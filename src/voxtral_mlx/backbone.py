"""Voxtral 4B TTS backbone — 26-layer Mistral causal transformer in MLX.

Uses Mistral-native weight naming so consolidated.safetensors loads with
one remap: mm_audio_embeddings.tok_embeddings.weight → tok_embeddings.weight.
"""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BackboneArgs:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131072
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5


class KVCache:
    """Pre-allocated KV cache with slice assignment.

    Matches mlx-lm's KVCache pattern: pre-allocates in chunks of `step`
    tokens, writes via slice assignment, returns a view up to the current
    offset. Avoids per-step array allocation and copy.
    """

    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class Attention(nn.Module):
    def __init__(self, args: BackboneArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.rope = nn.RoPE(args.head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: BackboneArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: BackboneArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attention(self.attention_norm(x), mask, cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Backbone(nn.Module):
    def __init__(self, args: BackboneArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        """Run the backbone. Returns hidden states [B, L, dim], NOT logits."""
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.tok_embeddings(input_ids)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = _create_mask(h, cache)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)

    def output_projection(self, h: mx.array) -> mx.array:
        """Tied embedding → logit projection. Separate from forward pass."""
        return self.tok_embeddings.as_linear(h)


def _create_mask(h: mx.array, cache: list) -> Optional[mx.array | str]:
    """Causal mask: None for single-token generation, 'causal' for prompt."""
    N = h.shape[1]
    if N == 1:
        return None
    return "causal"
