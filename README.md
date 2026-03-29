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

result = generate("Hello world.", voice="neutral_male", output_path="hello.wav")
```

Model weights (~8 GB) are downloaded automatically from Hugging Face on first use.

Voice embeddings ship as `.pt` files and are converted to MLX format on first use. This requires `torch` as a one-time dependency:

```bash
uv add torch
```

After conversion, torch is no longer needed.

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
    cfg_alpha=1.2,       # classifier-free guidance strength (default 1.2)
    max_frames=2000,     # safety limit on generation length
    model_path="mistralai/Voxtral-4B-TTS-2603",  # HF repo or local path
)

# Pre-load models to avoid reloading on every call
models = load_all_models()
result = generate("First sentence.", voice="neutral_male", models=models)
result = generate("Second sentence.", voice="neutral_male", models=models)
```

`generate()` returns a `GenerateResult` with:
- `path` — output WAV file path
- `n_frames` — number of audio frames generated
- `duration_seconds` — audio duration
- `timing` — dict with `prompt_ms`, `generation_ms`, `vocoder_ms`, `total_ms`, `backbone_avg_ms`, `acoustic_avg_ms`

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

Measured on M1 Max (64 GB). Benchmark: "The quick brown fox jumps over the lazy dog, and then proceeds to run across the meadow while the sun sets behind the distant hills." / `cheerful_female` / seed 42. 1 warmup + 5 trials, all SHA256-identical.

| Metric | Value |
|--------|-------|
| Audio duration | 9.12s |
| End-to-end | 8585ms |
| Real-time factor | 0.94x |
| Per-frame | 70.0ms (backbone 21.9ms + acoustic 47.5ms) |
| Frames | 114 |

Real-time factor < 1.0 means audio is generated faster than playback speed.

Reproduce with:

```bash
uv run python -m scripts.benchmark --model-path mistralai/Voxtral-4B-TTS-2603
```

## How it works

Three components ported from Mistral's [Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603):

1. **Backbone** — 26-layer Mistral causal transformer (3.4B params). Processes the text prompt with voice conditioning and generates hidden states autoregressively.
2. **Acoustic transformer** — 3-layer flow-matching transformer (394M params). Converts each hidden state into 37 audio codes (1 semantic + 36 acoustic) via 7-step Euler integration with classifier-free guidance.
3. **Vocoder** — Convolutional decoder with interleaved ALiBi transformers (151M params). Decodes audio codes into a 24 kHz waveform.

Each component was written from scratch in MLX and validated against Mistral's PyTorch reference implementation.

## Requirements

- Apple Silicon Mac (M1 or later)
- ~15 GB memory (model weights in BF16 + working memory)
- Python 3.11+
- `torch` for one-time voice embedding conversion (optional after first run)

## License

This code is MIT. Model weights are subject to [Mistral's CC-BY-NC-4.0 license](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

Built by redseaplume and hypomnematist.
