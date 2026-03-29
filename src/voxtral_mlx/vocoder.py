"""Voxtral 4B TTS vocoder — convolutional decoder with interleaved transformers in MLX.

Decodes discrete audio codes (1 semantic + 36 acoustic) into 24kHz waveform.
Architecture: CausalConv1d → [Transformer → CausalConvTranspose1d] × 3 → Transformer → CausalConv1d → unpatch.
All operations in NLC layout — no rearranges between blocks.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VocoderArgs:
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    norm_eps: float = 1e-2
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    use_biases: bool = False
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    causal: bool = True
    attn_sliding_window_size: int = 16
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    decoder_transformer_lengths: tuple[int, ...] = (2, 2, 2, 2)
    decoder_convs_kernels: tuple[int, ...] = (3, 4, 4, 4)
    decoder_convs_strides: tuple[int, ...] = (1, 2, 2, 2)


# --- Padding helpers (NLC layout) ---


def _pad_replicate(x: mx.array, left: int, right: int) -> mx.array:
    """Replicate (edge) padding along the L axis. x is [B, L, C]."""
    parts = []
    if left > 0:
        parts.append(mx.broadcast_to(x[:, :1, :], (x.shape[0], left, x.shape[2])))
    parts.append(x)
    if right > 0:
        parts.append(mx.broadcast_to(x[:, -1:, :], (x.shape[0], right, x.shape[2])))
    return mx.concatenate(parts, axis=1)


def _pad_reflect(x: mx.array, left: int, right: int) -> mx.array:
    """Reflect padding along the L axis. x is [B, L, C].

    Handles the edge case where input is shorter than the padding amount
    by zero-padding first (matching PyTorch's pad1d helper).
    """
    L = x.shape[1]
    max_pad = max(left, right)
    extra = 0
    if L <= max_pad:
        extra = max_pad - L + 1
        x = mx.pad(x, [(0, 0), (0, extra), (0, 0)])
        L = x.shape[1]

    parts = []
    if left > 0:
        parts.append(x[:, 1 : left + 1, :][:, ::-1, :])
    parts.append(x)
    if right > 0:
        parts.append(x[:, -(right + 1) : -1, :][:, ::-1, :])
    result = mx.concatenate(parts, axis=1)

    if extra > 0:
        result = result[:, : result.shape[1] - extra, :]
    return result


def pad1d(x: mx.array, left: int, right: int, mode: str = "reflect") -> mx.array:
    """Pad along the L axis of an [B, L, C] tensor."""
    if left == 0 and right == 0:
        return x
    if mode == "replicate":
        return _pad_replicate(x, left, right)
    elif mode == "reflect":
        return _pad_reflect(x, left, right)
    else:
        return mx.pad(x, [(0, 0), (left, right), (0, 0)])


# --- Convolution wrappers ---


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=bias,
        )
        self.pad_mode = pad_mode
        self._stride = stride
        self._effective_kernel_size = (kernel_size - 1) * dilation + 1
        self._padding_total = self._effective_kernel_size - stride

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, C] (NLC)
        L = x.shape[1]
        n_frames = (L - self._effective_kernel_size + self._padding_total) / self._stride + 1
        target_length = (math.ceil(n_frames) - 1) * self._stride + (self._effective_kernel_size - self._padding_total)
        extra_padding = int(target_length - L)
        x = pad1d(x, self._padding_total, extra_padding, mode=self.pad_mode)
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias,
        )
        self._kernel_size = kernel_size
        self._stride = stride

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, C] (NLC)
        out = self.conv(x)
        total_padding = self._kernel_size - self._stride
        right_trim = math.ceil(total_padding * 1.0)  # trim_ratio=1.0
        if right_trim > 0:
            out = out[:, : out.shape[1] - right_trim, :]
        return out


# --- Vocoder Attention (ALiBi + sliding window + QK norm) ---


def _get_alibi_slopes(n_heads: int) -> mx.array:
    """Geometric ALiBi slopes for n_heads (power of 2)."""
    r = 2.0 ** (-8.0 / n_heads)
    return mx.array([r**i for i in range(n_heads)], dtype=mx.float32)


class VocoderAttention(nn.Module):
    def __init__(self, args: VocoderArgs, window_size: int):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5
        self.window_size = window_size
        self.causal = args.causal

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)

        if args.qk_norm:
            self.q_norm = nn.RMSNorm(args.n_heads * args.head_dim, eps=args.qk_norm_eps)
            self.k_norm = nn.RMSNorm(args.n_kv_heads * args.head_dim, eps=args.qk_norm_eps)

        # Not a parameter — store as a frozen array that won't appear in parameters()
        self._alibi_slopes = _get_alibi_slopes(args.n_heads)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if hasattr(self, "q_norm"):
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        xk = xk.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        xv = xv.reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Build combined ALiBi + causal + sliding window mask [1, H, S, S]
        positions = mx.arange(S)
        rel_pos = positions[None, :] - positions[:, None]  # [S, S], rel_pos[i,j] = j - i

        slopes = self._alibi_slopes.astype(xq.dtype)
        attn_bias = slopes[:, None, None] * rel_pos[None, :, :].astype(xq.dtype)  # [H, S, S]

        neg_inf = mx.array(float("-inf"), dtype=xq.dtype)

        # Causal: mask future positions (j > i)
        if self.causal:
            attn_bias = mx.where(rel_pos[None, :, :] > 0, neg_inf, attn_bias)

        # Sliding window: mask beyond window
        window_left = self.window_size
        window_right = 0 if self.causal else self.window_size
        outside_window = (rel_pos < -window_left) | (rel_pos > window_right)
        attn_bias = mx.where(outside_window[None, :, :], neg_inf, attn_bias)

        mask = attn_bias[None, :, :, :]  # [1, H, S, S]

        output = mx.fast.scaled_dot_product_attention(
            xq, xk, xv, scale=self.scale, mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.wo(output)


# --- Vocoder FeedForward ---


class VocoderFeedForward(nn.Module):
    def __init__(self, args: VocoderArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=args.use_biases)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# --- Vocoder Transformer Block (with layer scale) ---


class VocoderTransformerBlock(nn.Module):
    def __init__(self, args: VocoderArgs, window_size: int):
        super().__init__()
        self.attention = VocoderAttention(args, window_size)
        self.feed_forward = VocoderFeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        self.has_layer_scale = args.layer_scale
        if self.has_layer_scale:
            init = args.layer_scale_init or 0.01
            self.attention_scale = mx.full((args.dim,), init)
            self.ffn_scale = mx.full((args.dim,), init)

    def __call__(self, x: mx.array) -> mx.array:
        r = self.attention(self.attention_norm(x))
        if self.has_layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.has_layer_scale:
            r = self.ffn_scale * r
        return h + r


# --- Vocoder Transformer (N blocks) ---


class VocoderTransformer(nn.Module):
    def __init__(self, args: VocoderArgs, n_layers: int, window_size: int):
        super().__init__()
        self.layers = [VocoderTransformerBlock(args, window_size) for _ in range(n_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


# --- Codebooks ---


class SemanticCodebook(nn.Module):
    """VQ codebook for semantic codes. Embedding is precomputed at load time."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        # Placeholder — replaced during weight loading with precomputed embedding
        self.embedding = mx.zeros((codebook_size, codebook_dim))

    def decode(self, codes: mx.array) -> mx.array:
        """codes: [B, T] int → [B, T, 256]"""
        return self.embedding[codes]


class AcousticCodebook:
    """FSQ decode — pure arithmetic, no learned parameters."""

    def __init__(self, n_levels: int = 21):
        self.n_levels = n_levels

    def decode(self, codes: mx.array) -> mx.array:
        """codes: [B, T, 36] int → [B, T, 36] float"""
        return (codes * 2.0 / (self.n_levels - 1)) - 1.0


class Codebook(nn.Module):
    """Combined semantic + acoustic codebook decode."""

    def __init__(self, args: VocoderArgs):
        super().__init__()
        self.semantic = SemanticCodebook(args.semantic_codebook_size, args.semantic_dim)
        self.acoustic = AcousticCodebook(args.acoustic_codebook_size)

    def decode(self, codes: mx.array) -> mx.array:
        """codes: [B, T, 37] int → [B, T, 292] float

        First column is semantic code, remaining 36 are acoustic.
        """
        semantic_codes = codes[:, :, 0]        # [B, T]
        acoustic_codes = codes[:, :, 1:]       # [B, T, 36]

        semantic_emb = self.semantic.decode(semantic_codes)   # [B, T, 256]
        acoustic_emb = self.acoustic.decode(acoustic_codes)   # [B, T, 36]

        return mx.concatenate([semantic_emb, acoustic_emb], axis=-1)  # [B, T, 292]


# --- Full Vocoder ---


class Vocoder(nn.Module):
    def __init__(self, args: VocoderArgs | None = None):
        super().__init__()
        if args is None:
            args = VocoderArgs()
        self.args = args
        self.patch_size = args.pretransform_patch_size
        latent_dim = args.semantic_dim + args.acoustic_dim  # 292

        self.codebook = Codebook(args)

        # Build decoder blocks — heterogeneous list
        decoder_blocks: list[nn.Module] = []

        # Block 0: CausalConv1d 292→1024, k=3, s=1, pad=replicate
        decoder_blocks.append(
            CausalConv1d(
                latent_dim, args.dim,
                kernel_size=args.decoder_convs_kernels[0],
                stride=args.decoder_convs_strides[0],
                pad_mode="replicate",
                bias=False,
            )
        )

        cur_window_size = 2  # Encoder's final window size after halving

        # Window doubles after each stride-2 ConvTranspose
        if args.decoder_convs_strides[0] > 1:
            cur_window_size *= 2

        for idx, n_layers in enumerate(args.decoder_transformer_lengths):
            # Transformer block
            decoder_blocks.append(
                VocoderTransformer(args, n_layers, cur_window_size)
            )
            # ConvTranspose (upsample) — except after the last transformer
            if (idx + 1) < len(args.decoder_transformer_lengths):
                stride = args.decoder_convs_strides[idx + 1]
                kernel = args.decoder_convs_kernels[idx + 1]
                if kernel != 1 or stride != 1:
                    decoder_blocks.append(
                        CausalConvTranspose1d(
                            args.dim, args.dim,
                            kernel_size=kernel,
                            stride=stride,
                            bias=False,
                        )
                    )
                    if stride > 1:
                        cur_window_size *= 2

        self.decoder_blocks = decoder_blocks

        # Output projection: CausalConv1d 1024→240, k=7, pad=reflect
        self.output_proj = CausalConv1d(
            args.dim, args.pretransform_patch_size,
            kernel_size=args.patch_proj_kernel_size,
            pad_mode="reflect",
            bias=False,
        )

    def __call__(self, codes: mx.array) -> mx.array:
        """Decode audio codes to waveform.

        Args:
            codes: [B, T, 37] int — 1 semantic + 36 acoustic codes per frame.

        Returns:
            waveform: [B, samples] float — 24kHz mono audio.
        """
        # Codebook decode → [B, T, 292]
        emb = self.codebook.decode(codes)

        # Decoder blocks — all NLC, no transposes
        for block in self.decoder_blocks:
            emb = block(emb)

        # Output projection → [B, 8T, 240]
        emb = self.output_proj(emb)

        # Unpatch → [B, 8T * 240]
        B = emb.shape[0]
        waveform = emb.reshape(B, -1)
        return waveform
