# Voxtral-4B-TTS-2603-MLX

[Mistral's Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

Text in, speech out. 20 voices, 9 languages. No cloud, no CUDA, just your Mac.

## Quickstart

Requires Apple Silicon and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/redseaplume/Voxtral-4B-TTS-2603-MLX.git
cd Voxtral-4B-TTS-2603-MLX
uv sync
```

```python
from voxtral_mlx import generate

result = generate("Hello world.", voice="cheerful_female", output_path="hello.wav")
```

Model weights (7.5 GB) are downloaded automatically from Hugging Face on first use.

Voice embeddings ship as `.pt` files and are converted to MLX format on first use. This requires `torch` as a one-time dependency:

```bash
uv add torch
```

After conversion, torch is no longer needed.

For a smaller, faster download that doesn't need torch at all, use the 4-bit quantized weights (3.4 GB):

```python
from voxtral_mlx import generate

result = generate("Hello world.", voice="cheerful_female", output_path="hello.wav",
                  model_path="redseaplume/Voxtral-4B-TTS-2603-MLX-4bit")
```

4-bit weights ship with pre-converted voice embedding. No torch required.

## API

```python
from voxtral_mlx import generate, load_all_models

# Simple
result = generate("Hello world.", voice="cheerful_female", output_path="hello.wav")

# With all options
result = generate(
    text="The sun sets behind the hills.",
    voice="cheerful_female",
    output_path="output.wav",
    seed=42,
    cfg_alpha=1.2,
    max_frames=2000,
    model_path="redseaplume/Voxtral-4B-TTS-2603-MLX-4bit",
)

# Pre-load models to avoid reloading on every call
models = load_all_models()
result = generate("First sentence.", voice="cheerful_female", models=models)
result = generate("Second sentence.", voice="cheerful_female", models=models)
```

`generate()` returns a `GenerateResult` with:
- `path`: output WAV file path
- `n_frames`: number of audio frames generated
- `duration_seconds`: audio duration
- `timing`: dict with `prompt_ms`, `generation_ms`, `vocoder_ms`, `total_ms`, `backbone_avg_ms`, `acoustic_avg_ms`

## Voices

20 preset voices across 9 languages:

| Language | Voices |
|----------|--------|
| English | `neutral_male`, `neutral_female`, `casual_male`, `casual_female`, `cheerful_female` |
| Spanish | `es_male`, `es_female` |
| French | `fr_male`, `fr_female` |
| German | `de_male`, `de_female` |
| Italian | `it_male`, `it_female` |
| Portuguese | `pt_male`, `pt_female` |
| Dutch | `nl_male`, `nl_female` |
| Arabic | `ar_male` |
| Hindi | `hi_male`, `hi_female` |

## Performance

Measured on M1 Max (64 GB). Benchmark text: "The quick brown fox jumps over the lazy dog, and then proceeds to run across the meadow while the sun sets behind the distant hills." / `cheerful_female` / seed 42. 1 warmup + 5 trials, all SHA256-identical.

**BF16 (default):**

| Metric | Value |
|--------|-------|
| Audio | 9.12s (114 frames) |
| End-to-end | 8971ms mean |
| Per-frame | 71.4ms (backbone 22.0ms + acoustic 48.8ms) |

**4-bit quantized backbone:**

| Metric | Value |
|--------|-------|
| Audio | 9.36s (117 frames) |
| End-to-end | 7861ms mean |
| Per-frame | 59.2ms (backbone 9.1ms + acoustic 49.4ms) |

4-bit quantization reduces the backbone weights from 6.4 GB to 2.5 GB. Audio quality is slightly different; the generation trajectory diverges, producing a few extra frames. Acoustic transformer and vocoder are unchanged.

Reproduce with:

```bash
uv run python -m scripts.benchmark
uv run python -m scripts.benchmark --model-path redseaplume/Voxtral-4B-TTS-2603-MLX-4bit
```

## How it works

Three model components: backbone, acoustic transformer, and vocoder, ported from scratch to MLX. Tokenization uses Mistral's `mistral-common` library.

1. **Backbone**: 26-layer Mistral causal transformer (3.4B params). Processes the text prompt with voice conditioning and generates hidden states autoregressively.
2. **Acoustic transformer**: 3-layer flow-matching transformer (394M params). Converts each hidden state into 37 audio codes (1 semantic + 36 acoustic) via 7-step Euler integration with classifier-free guidance.
3. **Vocoder**: Convolutional decoder with interleaved ALiBi transformers (152M params). Decodes audio codes into a 24 kHz waveform.

Validated against Mistral's PyTorch reference implementation at each stage.

## Compared to mlx-audio

[mlx-audio](https://github.com/Blaizzy/mlx-audio) is an excellent multi-model audio library that also supports Voxtral TTS. Voxtral-4B-TTS-2603-MLX is a standalone, single-model port with different engineering tradeoffs:

| | **This project** | **mlx-audio** |
|---|---|---|
| Backbone | From scratch | mlx-lm LlamaModel wrapper |
| Tokenizer | `mistral-common` (official) | Custom fallback (official optional) |
| CFG passes | Batched (single forward call) | Sequential (two calls per step) |
| PRNG | Explicit key per frame (deterministic) | Global random state |
| Eval strategy | `mx.eval()` per frame | Lazy (no per-frame eval) |

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.11+
- BF16 path: `torch` for one-time voice embedding conversion
- 4-bit path: no torch needed

## License

This code is MIT. Model weights are subject to [Mistral's CC-BY-NC-4.0 license](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

Built by redseaplume and hypomnematist.
