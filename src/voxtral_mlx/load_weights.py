"""Load all model weights from consolidated.safetensors."""

import logging
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .acoustic_transformer import AcousticTransformerArgs, FlowMatchingAcousticTransformer
from .backbone import Backbone, BackboneArgs
from .vocoder import Vocoder, VocoderArgs

logger = logging.getLogger("voxtral_mlx")

DEFAULT_MODEL = "mistralai/Voxtral-4B-TTS-2603"

# Prefixes that belong to the backbone
BACKBONE_PREFIXES = ("layers.", "norm.")
# The one remap
EMBEDDING_REMAP = {
    "mm_audio_embeddings.tok_embeddings.weight": "tok_embeddings.weight",
}

VOCODER_PREFIX = "audio_tokenizer."
ACOUSTIC_TRANSFORMER_PREFIX = "acoustic_transformer."


def resolve_model_path(path_or_repo: str = DEFAULT_MODEL) -> Path:
    """Resolve a local path or HuggingFace repo ID to a local directory.

    If the path exists locally, returns it directly. Otherwise downloads
    from HuggingFace Hub and returns the cached path.

    Args:
        path_or_repo: Local directory path or HuggingFace repo ID
            (e.g. "mistralai/Voxtral-4B-TTS-2603").

    Returns:
        Path to a local directory containing model files.
    """
    path = Path(path_or_repo)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    logger.info("Downloading %s from HuggingFace Hub...", path_or_repo)
    local_dir = snapshot_download(
        path_or_repo,
        allow_patterns=[
            "consolidated.safetensors",
            "params.json",
            "tekken.json",
            "voice_embedding/*",
        ],
    )
    return Path(local_dir)


def _find_weights_file(model_path: Path) -> Path:
    """Find the safetensors weights file in a model directory."""
    # Direct path to a file
    if model_path.is_file():
        return model_path
    # Standard name
    consolidated = model_path / "consolidated.safetensors"
    if consolidated.exists():
        return consolidated
    raise FileNotFoundError(
        f"No consolidated.safetensors found in {model_path}"
    )


def load_all_weights(model_path: Path) -> dict[str, mx.array]:
    """Load the safetensors file once, return the full weight dict."""
    weights_file = _find_weights_file(model_path)
    return mx.load(str(weights_file))


def load_backbone(
    all_weights: dict[str, mx.array],
    args: BackboneArgs | None = None,
) -> Backbone:
    """Instantiate the backbone and load weights from a pre-loaded weight dict."""
    if args is None:
        args = BackboneArgs()

    model = Backbone(args)

    backbone_weights = {}
    for key, value in all_weights.items():
        if key in EMBEDDING_REMAP:
            backbone_weights[EMBEDDING_REMAP[key]] = value
            continue
        if any(key.startswith(p) for p in BACKBONE_PREFIXES):
            backbone_weights[key] = value

    model.load_weights(list(backbone_weights.items()))
    return model


def _reconstruct_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    """Reconstruct weight from weight norm parametrization: weight = g * (v / ||v||).

    g: [dim0, 1, 1] — magnitude
    v: [dim0, dim1, dim2] — direction
    Returns: [dim0, dim1, dim2]
    """
    v_flat = v.reshape(v.shape[0], -1)
    v_norm = mx.linalg.norm(v_flat, axis=1, keepdims=True)[..., None]  # [dim0, 1, 1]
    return g * (v / v_norm)


def _is_conv1d_weight(base_key: str) -> bool:
    """Check if base_key belongs to a Conv1d (decoder_blocks.0 or output_proj)."""
    return base_key in ("decoder_blocks.0.conv", "output_proj.conv")


def _is_conv_transpose_weight(base_key: str) -> bool:
    """Check if base_key belongs to a ConvTranspose1d (decoder_blocks.2,4,6)."""
    m = re.match(r"decoder_blocks\.(\d+)\.conv", base_key)
    return m is not None and int(m.group(1)) in (2, 4, 6)


def _process_vocoder_weights(all_weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Extract and process vocoder weights from the full weight dict."""
    vocoder_raw: dict[str, mx.array] = {}
    for key, value in all_weights.items():
        if key.startswith(VOCODER_PREFIX):
            vocoder_raw[key[len(VOCODER_PREFIX):]] = value

    vocoder_weights: dict[str, mx.array] = {}
    wn_pairs: dict[str, dict[str, mx.array]] = {}
    cluster_usage = None
    embedding_sum = None

    for key, value in vocoder_raw.items():
        if ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            wn_pairs.setdefault(base, {})["original0"] = value
            continue
        if ".parametrizations.weight.original1" in key:
            base = key.replace(".parametrizations.weight.original1", "")
            wn_pairs.setdefault(base, {})["original1"] = value
            continue
        if key == "quantizer.semantic_codebook.cluster_usage":
            cluster_usage = value
            continue
        if key == "quantizer.semantic_codebook.embedding_sum":
            embedding_sum = value
            continue
        vocoder_weights[key] = value

    for base_key, pair in wn_pairs.items():
        g = pair["original0"]
        v = pair["original1"]
        weight = _reconstruct_weight_norm(g, v)
        if _is_conv1d_weight(base_key):
            weight = weight.swapaxes(1, 2)
        elif _is_conv_transpose_weight(base_key):
            weight = weight.transpose(1, 2, 0)
        vocoder_weights[base_key + ".weight"] = weight

    if cluster_usage is not None and embedding_sum is not None:
        embedding = embedding_sum / mx.maximum(cluster_usage[:, None], 1e-5)
        vocoder_weights["codebook.semantic.embedding"] = embedding

    return vocoder_weights


def load_vocoder(
    all_weights: dict[str, mx.array],
    args: VocoderArgs | None = None,
) -> Vocoder:
    """Instantiate the vocoder and load weights from a pre-loaded weight dict."""
    if args is None:
        args = VocoderArgs()

    model = Vocoder(args)
    vocoder_weights = _process_vocoder_weights(all_weights)
    model.load_weights(list(vocoder_weights.items()))
    return model


def load_acoustic_transformer(
    all_weights: dict[str, mx.array],
    args: AcousticTransformerArgs | None = None,
) -> FlowMatchingAcousticTransformer:
    """Instantiate the acoustic transformer and load weights from a pre-loaded weight dict."""
    if args is None:
        args = AcousticTransformerArgs()

    model = FlowMatchingAcousticTransformer(args)

    at_weights: dict[str, mx.array] = {}
    for key, value in all_weights.items():
        if key.startswith(ACOUSTIC_TRANSFORMER_PREFIX):
            at_weights[key[len(ACOUSTIC_TRANSFORMER_PREFIX):]] = value

    model.load_weights(list(at_weights.items()))
    return model
