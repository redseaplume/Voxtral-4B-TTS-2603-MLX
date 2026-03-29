"""Voxtral 4B TTS acoustic transformer — flow-matching 3-layer bidirectional transformer in MLX.

Converts backbone hidden states into audio codes (1 semantic + 36 acoustic) per frame.
Uses Euler integration with classifier-free guidance for acoustic code prediction.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AcousticTransformerArgs:
    input_dim: int = 3072
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    use_biases: bool = False
    norm_eps: float = 1e-5
    # Flow matching
    n_acoustic_codebooks: int = 36
    acoustic_codebook_size: int = 21
    semantic_codebook_size: int = 8192
    semantic_output_size: int = 8320  # round_up_to_128(8192 + 2)
    noise_scale: float = 1.0
    n_euler_steps: int = 8
    cfg_alpha: float = 1.2


# --- Time Embedding ---


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for encoding flow-matching timestep."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        half_dim = dim // 2
        # inv_freq is NOT in the checkpoint — computed here
        self._inv_freq = mx.exp(
            -math.log(theta) * mx.arange(half_dim).astype(mx.float32) / half_dim
        )

    def __call__(self, t: mx.array) -> mx.array:
        """t: [B, 1] float → [B, dim]"""
        emb = t * self._inv_freq  # [B, half_dim]
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


# --- Bidirectional Attention (no ALiBi, no RoPE, no QK norm, no causal mask) ---


class BidirectionalAttention(nn.Module):
    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = args.n_heads // args.n_kv_heads
        self.scale = args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        xk = xk.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        xv = xv.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            xq, xk, xv, scale=self.scale,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.wo(output)


# --- FeedForward (SwiGLU, same as backbone) ---


class FeedForward(nn.Module):
    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=args.use_biases)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# --- Transformer Block ---


class AcousticTransformerBlock(nn.Module):
    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.attention = BidirectionalAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# --- Flow Matching Acoustic Transformer ---


# Special token IDs (same as AudioSpecialTokens enum in PyTorch reference)
EMPTY_AUDIO_ID = 0
END_AUDIO_ID = 1
N_SPECIAL_TOKENS = 2


class FlowMatchingAcousticTransformer(nn.Module):
    def __init__(self, args: AcousticTransformerArgs | None = None):
        super().__init__()
        if args is None:
            args = AcousticTransformerArgs()
        self.args = args

        # Time embedding — inv_freq computed, not loaded
        self.time_embedding = TimeEmbedding(args.dim)

        # Projections
        self.input_projection = nn.Linear(args.n_acoustic_codebooks, args.dim, bias=False)
        self.time_projection = nn.Linear(args.dim, args.dim, bias=False)
        self.llm_projection = nn.Linear(args.input_dim, args.dim, bias=False)

        # Transformer layers
        self.layers = [AcousticTransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        # Output heads
        self.semantic_codebook_output = nn.Linear(args.dim, args.semantic_output_size, bias=False)
        self.acoustic_codebook_output = nn.Linear(args.dim, args.n_acoustic_codebooks, bias=False)

    def _predict_velocity(
        self,
        x_t: mx.array,       # [B, 36]
        llm_output: mx.array,  # [B, 3072]
        t_emb: mx.array,      # [B, 3072]
    ) -> mx.array:
        """Predict velocity for Euler step. Returns [B, 36]."""
        x_t = x_t.astype(llm_output.dtype)

        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)

        # Build 3-token sequence: [acoustic, time, hidden]
        seq = mx.concatenate([
            self.input_projection(x_t[:, None, :]),  # [B, 1, 3072]
            t_emb[:, None, :],                         # [B, 1, 3072]
            llm_output[:, None, :],                    # [B, 1, 3072]
        ], axis=1)  # [B, 3, 3072]

        # Run transformer
        h = seq
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Velocity from position 0
        v_t = self.acoustic_codebook_output(h[:, 0, :])  # [B, 36]
        return v_t

    def decode_one_frame(
        self,
        semantic_code: mx.array,   # [B]
        llm_hidden: mx.array,      # [B, 3072]
        key: mx.array | None = None,
    ) -> mx.array:
        """Run Euler loop to produce 36 acoustic codes. Returns [B, 36] int."""
        B = semantic_code.shape[0]
        args = self.args

        # Skip if END_AUDIO
        should_decode = semantic_code != END_AUDIO_ID

        # Start from noise
        if key is None:
            key = mx.random.key(0)
        x_0 = args.noise_scale * mx.random.normal(shape=(B, args.n_acoustic_codebooks), key=key)
        x_0 = x_0.astype(llm_hidden.dtype)

        # Timesteps
        timesteps = mx.linspace(0.0, 1.0, args.n_euler_steps).astype(llm_hidden.dtype)
        llm_hidden_zero = mx.zeros_like(llm_hidden)

        # Euler integration
        sampled = x_0
        for i in range(args.n_euler_steps - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = self.time_embedding(
                mx.broadcast_to(t.reshape(1, 1), (B, 1))
            ).astype(llm_hidden.dtype)  # [B, 3072]

            # Batch cond + uncond
            x_batched = mx.concatenate([sampled, sampled], axis=0)
            llm_batched = mx.concatenate([llm_hidden, llm_hidden_zero], axis=0)
            t_emb_batched = mx.concatenate([t_emb, t_emb], axis=0)

            v_all = self._predict_velocity(x_batched, llm_batched, t_emb_batched)
            v_cond = v_all[:B]
            v_uncond = v_all[B:]
            v_t = args.cfg_alpha * v_cond + (1 - args.cfg_alpha) * v_uncond

            sampled = sampled + v_t * dt

        # Quantize
        sampled = mx.clip(sampled, -1.0, 1.0)
        scaled = ((sampled + 1.0) / 2.0) * (args.acoustic_codebook_size - 1)
        output_codes = mx.round(scaled).astype(mx.int32)

        # Mask END_AUDIO frames
        output_codes = mx.where(
            should_decode[:, None],
            output_codes,
            mx.zeros_like(output_codes),  # EMPTY_AUDIO_ID = 0
        )

        # Add special token offset
        return output_codes + N_SPECIAL_TOKENS

    def __call__(
        self,
        llm_hidden: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        """Predict audio codes for one frame.

        Args:
            llm_hidden: [B, 3072] backbone hidden state
            key: optional PRNG key for reproducible noise

        Returns:
            audio_codes: [B, 37] int — 1 semantic + 36 acoustic
        """
        # Semantic code: argmax with masking
        semantic_logit = self.semantic_codebook_output(llm_hidden).astype(mx.float32)
        semantic_logit = semantic_logit.at[:, EMPTY_AUDIO_ID].add(-float("inf"))
        semantic_logit = semantic_logit.at[
            :, (N_SPECIAL_TOKENS + self.args.semantic_codebook_size):
        ].add(-float("inf"))

        semantic_code = mx.argmax(semantic_logit, axis=-1, keepdims=True)  # [B, 1]

        # Acoustic codes via flow matching
        acoustic_codes = self.decode_one_frame(
            semantic_code.squeeze(1),
            llm_hidden,
            key=key,
        )  # [B, 36]

        # Concatenate: [semantic, acoustic]
        return mx.concatenate([semantic_code, acoustic_codes], axis=1)  # [B, 37]
