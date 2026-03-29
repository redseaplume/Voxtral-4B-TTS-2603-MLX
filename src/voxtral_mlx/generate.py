"""Voxtral 4B TTS generation — end-to-end text-to-speech in MLX.

Handles:
- Voice embedding injection: replace [AUDIO] token positions with voice embedding rows
- Audio codebook feedback: offset-shifted embedding lookup + sum over 37 codebooks
- Full generation loop: tokenize → embed → backbone → acoustic transformer → vocoder → WAV
- Weight loading for all components
"""

import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mistral_common.protocol.speech.request import SpeechRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from .acoustic_transformer import END_AUDIO_ID, FlowMatchingAcousticTransformer
from .backbone import Backbone, KVCache
from .load_weights import (
    DEFAULT_MODEL,
    load_acoustic_transformer,
    load_all_weights,
    load_backbone,
    load_vocoder,
    resolve_model_path,
)
from .vocoder import Vocoder

logger = logging.getLogger("voxtral_mlx")


@dataclass
class GenerateResult:
    """Result of a generation run."""

    path: str
    n_frames: int
    duration_seconds: float
    sample_rate: int = 24000
    timing: dict[str, float] = field(default_factory=dict)

# Special token IDs (from tekken.json)
AUDIO_TOKEN_ID = 24

# Audio codebook constants
N_CODEBOOKS = 37
N_SPECIAL_TOKENS = 2  # EMPTY_AUDIO=0, END_AUDIO=1
SEMANTIC_CODEBOOK_SIZE = 8192
ACOUSTIC_CODEBOOK_SIZE = 21


def _compute_codebook_offsets() -> mx.array:
    """Compute per-codebook offsets for the shared embedding table.

    Sizes (with special tokens, no padding):
      semantic: 8192 + 2 = 8194
      acoustic (x36): 21 + 2 = 23 each
    Offsets: cumsum([0] + sizes[:-1])
    """
    sizes = [SEMANTIC_CODEBOOK_SIZE + N_SPECIAL_TOKENS] + [
        ACOUSTIC_CODEBOOK_SIZE + N_SPECIAL_TOKENS
    ] * (N_CODEBOOKS - 1)
    offsets = np.cumsum([0] + sizes[:-1]).astype(np.int32)
    return mx.array(offsets)


class AudioCodebookEmbedding(nn.Module):
    """Offset-shifted multi-codebook embedding (MultiVocabEmbeddings in PyTorch reference).

    Single embedding table [9088, 3072]. Each of the 37 codebooks indexes into
    a different offset range. Forward: offset-shift codes, look up, sum over codebooks.
    """

    def __init__(self, dim: int = 3072):
        super().__init__()
        # Total vocab: 8194 + 36*23 = 9022, padded to 128 → 9088
        total_vocab = SEMANTIC_CODEBOOK_SIZE + N_SPECIAL_TOKENS + (N_CODEBOOKS - 1) * (ACOUSTIC_CODEBOOK_SIZE + N_SPECIAL_TOKENS)
        padded_size = 128 * ((total_vocab + 127) // 128)
        self.embeddings = nn.Embedding(padded_size, dim)
        self._offsets = _compute_codebook_offsets()

    def __call__(self, codes: mx.array) -> mx.array:
        """Look up and sum codebook embeddings.

        Args:
            codes: [B, 37] int — 1 semantic + 36 acoustic codes per frame.
                   Codes already include +2 special token offset.

        Returns:
            [B, 3072] — summed embedding across all 37 codebooks.
        """
        # [B, 37] + [37] → [B, 37] offset-shifted indices
        shifted = codes + self._offsets
        # [B, 37, 3072]
        emb = self.embeddings(shifted)
        # [B, 3072] — sum over codebook dimension
        return emb.sum(axis=1)


def inject_voice_embedding(
    backbone: nn.Module,
    input_ids: mx.array,
    voice_embedding: mx.array,
) -> mx.array:
    """Build prompt embeddings with voice embedding injection.

    Runs tok_embeddings on input_ids, then replaces positions where
    input_ids == AUDIO_TOKEN_ID (24) with voice embedding rows in order.

    The AUDIO tokens are contiguous in the prompt:
    [BOS, BEGIN_AUDIO, AUDIO*N, NEXT_AUDIO_TEXT, text..., REPEAT_AUDIO_TEXT, BEGIN_AUDIO]

    Args:
        backbone: The backbone model (has tok_embeddings).
        input_ids: [1, L] int token IDs.
        voice_embedding: [N_frames, 3072] BF16 voice embedding.

    Returns:
        [1, L, 3072] embeddings with voice rows injected.
    """
    embeddings = backbone.tok_embeddings(input_ids)  # [1, L, 3072]

    # AUDIO tokens start at position 2 (after BOS, BEGIN_AUDIO)
    n_voice_frames = voice_embedding.shape[0]
    start = 2
    end = start + n_voice_frames

    # Verify the positions are actually AUDIO tokens
    ids_flat = input_ids[0]
    audio_slice = ids_flat[start:end]
    mx.eval(audio_slice)
    assert (audio_slice == AUDIO_TOKEN_ID).all().item(), (
        f"Expected all AUDIO tokens at positions {start}:{end}"
    )

    # Replace with voice embedding rows
    embeddings[0, start:end] = voice_embedding.astype(embeddings.dtype)

    return embeddings


def _load_audio_codebook_embedding(
    all_weights: dict[str, mx.array],
    dim: int = 3072,
) -> AudioCodebookEmbedding:
    """Load the audio codebook embedding table from a pre-loaded weight dict."""
    model = AudioCodebookEmbedding(dim)

    key = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
    assert key in all_weights, f"Key {key!r} not found in weights"

    model.load_weights([("embeddings.weight", all_weights[key])])
    return model


# --- Tokenization ---


def tokenize(text: str, voice: str, tekken_path: Path) -> list[int]:
    """Tokenize text + voice into the TTS prompt token sequence."""
    tok = MistralTokenizer.from_file(str(tekken_path))
    result = tok.encode_speech_request(SpeechRequest(input=text, voice=voice))
    return result.tokens


def _convert_voice_embedding(pt_path: Path, npz_path: Path) -> None:
    """Convert a .pt voice embedding to .npz format (requires torch)."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"Voice embedding {pt_path.name} needs conversion from .pt to .npz. "
            f"Install torch to convert automatically: uv add torch\n"
            f"Or run: uv run --extra voice-convert python -c "
            f"'from voxtral_mlx.generate import _convert_voice_embedding; "
            f"_convert_voice_embedding(\"{pt_path}\", \"{npz_path}\")'"
        )

    data = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    # Lossless BF16 transfer via uint16 view
    np_uint16 = data.view(torch.uint16).numpy()
    embedding = mx.array(np_uint16).view(mx.bfloat16)
    mx.savez(str(npz_path), embedding=embedding)
    logger.info("Converted %s → %s", pt_path.name, npz_path.name)


def load_voice_embedding(voice: str, voice_dir: Path) -> mx.array:
    """Load a voice embedding by name, converting from .pt if needed."""
    npz_path = voice_dir / f"{voice}.npz"
    if npz_path.exists():
        return mx.load(str(npz_path))["embedding"]

    # Try converting from .pt
    pt_path = voice_dir / f"{voice}.pt"
    if pt_path.exists():
        _convert_voice_embedding(pt_path, npz_path)
        return mx.load(str(npz_path))["embedding"]

    available = sorted(
        p.stem for p in voice_dir.glob("*.npz")
    ) or sorted(
        p.stem for p in voice_dir.glob("*.pt")
    )
    raise FileNotFoundError(
        f"Voice embedding not found: {voice}\n"
        f"Available voices: {', '.join(available)}"
    )


# --- WAV output ---


def save_wav(waveform: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Save float waveform as 16-bit mono WAV."""
    # Clip and convert to int16
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)

    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


# --- Full generation ---


def load_all_models(
    model_path: str | Path = DEFAULT_MODEL,
) -> tuple[Backbone, FlowMatchingAcousticTransformer, Vocoder, AudioCodebookEmbedding]:
    """Load all four model components.

    Args:
        model_path: Local directory path or HuggingFace repo ID
            (e.g. "mistralai/Voxtral-4B-TTS-2603"). Weights are downloaded
            and cached automatically if not found locally.

    Returns:
        Tuple of (backbone, acoustic_transformer, vocoder, codebook_embedding).
    """
    resolved = resolve_model_path(str(model_path))
    all_weights = load_all_weights(resolved)

    backbone = load_backbone(all_weights)
    acoustic_transformer = load_acoustic_transformer(all_weights)
    vocoder = load_vocoder(all_weights)
    codebook_emb = _load_audio_codebook_embedding(all_weights)
    return backbone, acoustic_transformer, vocoder, codebook_emb


def generate(
    text: str,
    voice: str = "neutral_male",
    output_path: str = "output.wav",
    seed: int = 0,
    cfg_alpha: float | None = None,
    max_frames: int = 2000,
    model_path: str | Path = DEFAULT_MODEL,
    models: tuple | None = None,
) -> GenerateResult:
    """Generate speech from text.

    Args:
        text: Input text to speak.
        voice: Voice preset name (e.g. "neutral_male", "cheerful_female").
            Available voices: casual_female, casual_male, cheerful_female,
            neutral_female, neutral_male, fr_male, fr_female, es_male,
            es_female, de_male, de_female, it_male, it_female, pt_male,
            pt_female, nl_male, nl_female, ar_male, hi_male, hi_female.
        output_path: Where to save the WAV file.
        seed: PRNG seed for reproducible generation.
        cfg_alpha: Classifier-free guidance strength. Default 1.2 (matches
            Mistral's reference). Lower values may sound more natural.
            Higher values adhere more to the voice prompt.
        max_frames: Maximum number of audio frames to generate.
        model_path: Local directory path or HuggingFace repo ID. Weights are
            downloaded and cached automatically if not found locally.
        models: Optional pre-loaded models from load_all_models(). Pass this
            to avoid reloading weights on every call.

    Returns:
        GenerateResult with path, timing, and metadata.
    """
    t_start = time.time()

    # Resolve model path
    resolved = resolve_model_path(str(model_path))

    # Load models
    if models is not None:
        backbone, acoustic_transformer, vocoder, codebook_emb = models
    else:
        logger.info("Loading models...")
        all_weights = load_all_weights(resolved)
        backbone = load_backbone(all_weights)
        acoustic_transformer = load_acoustic_transformer(all_weights)
        vocoder = load_vocoder(all_weights)
        codebook_emb = _load_audio_codebook_embedding(all_weights)

    # Set CFG alpha if provided
    if cfg_alpha is not None:
        acoustic_transformer.args.cfg_alpha = cfg_alpha

    # 1. Tokenize
    tekken_path = resolved / "tekken.json"
    token_ids = tokenize(text, voice, tekken_path)
    input_ids = mx.array([token_ids])
    logger.info("Prompt: %d tokens", len(token_ids))

    # 2. Build embeddings with voice injection
    voice_dir = resolved / "voice_embedding"
    voice_embedding = load_voice_embedding(voice, voice_dir)
    prompt_embeddings = inject_voice_embedding(backbone, input_ids, voice_embedding)

    # 3. Feed prompt through backbone with KV cache
    t_prompt_start = time.time()
    cache = [KVCache() for _ in range(backbone.args.n_layers)]
    hidden = backbone(input_embeddings=prompt_embeddings, cache=cache)
    mx.eval(hidden)
    t_prompt_end = time.time()

    # Last position hidden state
    last_hidden = hidden[:, -1, :]  # [1, 3072]

    # 4. Generation loop
    t_gen_start = time.time()
    key = mx.random.key(seed)
    all_codes = []
    backbone_times = []
    acoustic_times = []

    for frame_idx in range(max_frames):
        # Split PRNG key
        key, subkey = mx.random.split(key)

        # Acoustic transformer: hidden → codes
        t_ac_start = time.time()
        codes = acoustic_transformer(last_hidden, key=subkey)  # [1, 37]
        mx.eval(codes)
        t_ac_end = time.time()
        acoustic_times.append(t_ac_end - t_ac_start)

        # Check for END_AUDIO
        semantic_code = codes[0, 0].item()
        if semantic_code == END_AUDIO_ID:
            break

        all_codes.append(codes)

        # Audio codebook embedding: codes → embedding for next backbone step
        code_embedding = codebook_emb(codes)  # [1, 3072]
        code_embedding = code_embedding[:, None, :]  # [1, 1, 3072]

        # Feed back through backbone
        t_bb_start = time.time()
        hidden = backbone(input_embeddings=code_embedding, cache=cache)
        mx.eval(hidden)
        t_bb_end = time.time()
        backbone_times.append(t_bb_end - t_bb_start)
        last_hidden = hidden[:, -1, :]  # [1, 3072]

        if (frame_idx + 1) % 50 == 0:
            logger.info("  %d frames...", frame_idx + 1)

    t_gen_end = time.time()

    n_frames = len(all_codes)
    logger.info("Generated %d frames", n_frames)

    if backbone_times:
        avg_bb = sum(backbone_times) / len(backbone_times) * 1000
        avg_ac = sum(acoustic_times[:len(backbone_times)]) / len(backbone_times) * 1000
        logger.info(
            "Per-frame avg: backbone %.1fms, acoustic %.1fms, total %.1fms",
            avg_bb, avg_ac, avg_bb + avg_ac,
        )

    if n_frames == 0:
        logger.warning("No audio frames generated (immediate END_AUDIO)")
        return GenerateResult(
            path=output_path,
            n_frames=0,
            duration_seconds=0.0,
            timing={
                "prompt_ms": (t_prompt_end - t_prompt_start) * 1000,
                "generation_ms": (t_gen_end - t_gen_start) * 1000,
                "vocoder_ms": 0.0,
                "total_ms": (time.time() - t_start) * 1000,
            },
        )

    # 5. Post-process: stack codes, strip special token offset
    stacked = mx.concatenate(all_codes, axis=0)  # [n_frames, 37]
    stacked = stacked - N_SPECIAL_TOKENS  # strip +2 offset
    mx.eval(stacked)

    # Reshape for vocoder: [1, n_frames, 37]
    vocoder_input = stacked[None, :, :]

    # 6. Vocoder → waveform
    t_vocoder_start = time.time()
    waveform = vocoder(vocoder_input)  # [1, samples]
    mx.eval(waveform)
    t_vocoder_end = time.time()

    waveform_np = np.array(waveform[0].astype(mx.float32))
    duration_seconds = len(waveform_np) / 24000

    # 7. Save WAV
    save_wav(waveform_np, output_path)

    t_end = time.time()

    timing = {
        "prompt_ms": (t_prompt_end - t_prompt_start) * 1000,
        "generation_ms": (t_gen_end - t_gen_start) * 1000,
        "vocoder_ms": (t_vocoder_end - t_vocoder_start) * 1000,
        "total_ms": (t_end - t_start) * 1000,
    }
    if backbone_times:
        timing["backbone_avg_ms"] = sum(backbone_times) / len(backbone_times) * 1000
        timing["acoustic_avg_ms"] = sum(acoustic_times[:len(backbone_times)]) / len(backbone_times) * 1000

    logger.info(
        "Done: %d frames, %.2fs audio, %.1fs total (prompt %.1fs, generation %.1fs, vocoder %.1fs)",
        n_frames,
        duration_seconds,
        timing["total_ms"] / 1000,
        timing["prompt_ms"] / 1000,
        timing["generation_ms"] / 1000,
        timing["vocoder_ms"] / 1000,
    )

    return GenerateResult(
        path=output_path,
        n_frames=n_frames,
        duration_seconds=duration_seconds,
        timing=timing,
    )
