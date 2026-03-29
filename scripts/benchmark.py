"""Benchmark harness for Voxtral MLX generation.

Loads models once, runs warmup, then N timed trials.
Reports end-to-end time, generation loop time, per-frame average, and SHA256 correctness.

Usage:
    uv run python -m scripts.benchmark             # BF16 baseline
    uv run python -m scripts.benchmark --quantize   # 4-bit backbone quantization
"""

import argparse
import hashlib
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from voxtral_mlx.generate import generate, load_all_models

TEXT = "The quick brown fox jumps over the lazy dog, and then proceeds to run across the meadow while the sun sets behind the distant hills."
VOICE = "cheerful_female"
SEED = 42
EXPECTED_FRAMES = 114
N_TRIALS = 5
OUTPUT_PATH = "/tmp/benchmark_output.wav"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_once(models) -> dict:
    t0 = time.perf_counter()
    result = generate(
        text=TEXT,
        voice=VOICE,
        seed=SEED,
        output_path=OUTPUT_PATH,
        models=models,
    )
    t1 = time.perf_counter()

    return {
        "e2e_ms": (t1 - t0) * 1000,
        "gen_ms": result.timing["generation_ms"],
        "per_frame_ms": result.timing["generation_ms"] / result.n_frames,
        "backbone_avg_ms": result.timing.get("backbone_avg_ms", 0),
        "acoustic_avg_ms": result.timing.get("acoustic_avg_ms", 0),
        "n_frames": result.n_frames,
        "sha256": sha256_file(OUTPUT_PATH),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true", help="4-bit quantize the backbone")
    parser.add_argument("--model-path", default=None, help="Local path or HF repo ID")
    args = parser.parse_args()

    print("Loading models...")
    models = load_all_models(args.model_path) if args.model_path else load_all_models()
    backbone = models[0]

    if args.quantize:
        print("Quantizing backbone (4-bit, group_size=64)...")
        nn.quantize(backbone, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear))

    mx.eval(backbone.parameters())  # ensure weights are materialized

    print("Warmup run (discarded)...")
    warmup = run_once(models)
    print(f"  warmup: {warmup['e2e_ms']:.0f}ms, {warmup['n_frames']} frames, sha256={warmup['sha256'][:16]}...")

    if warmup["n_frames"] != EXPECTED_FRAMES:
        print(f"  WARNING: expected {EXPECTED_FRAMES} frames, got {warmup['n_frames']}")

    print(f"\nRunning {N_TRIALS} trials...")
    results = []
    for i in range(N_TRIALS):
        r = run_once(models)
        results.append(r)
        print(f"  trial {i+1}: e2e={r['e2e_ms']:.0f}ms  gen={r['gen_ms']:.0f}ms  per_frame={r['per_frame_ms']:.1f}ms  backbone={r['backbone_avg_ms']:.1f}ms  acoustic={r['acoustic_avg_ms']:.1f}ms  sha256={r['sha256'][:16]}...")

    # Correctness check
    hashes = set(r["sha256"] for r in results)
    all_match = len(hashes) == 1
    warmup_matches = warmup["sha256"] in hashes

    # Stats
    e2e = [r["e2e_ms"] for r in results]
    gen = [r["gen_ms"] for r in results]
    pf = [r["per_frame_ms"] for r in results]
    bb = [r["backbone_avg_ms"] for r in results]
    ac = [r["acoustic_avg_ms"] for r in results]

    print(f"\n--- Results ({N_TRIALS} trials) ---")
    print(f"End-to-end:     mean={statistics.mean(e2e):.0f}ms  std={statistics.stdev(e2e):.0f}ms  min={min(e2e):.0f}ms  max={max(e2e):.0f}ms")
    print(f"Generation:     mean={statistics.mean(gen):.0f}ms  std={statistics.stdev(gen):.0f}ms  min={min(gen):.0f}ms  max={max(gen):.0f}ms")
    print(f"Per-frame:      mean={statistics.mean(pf):.1f}ms  std={statistics.stdev(pf):.1f}ms  min={min(pf):.1f}ms  max={max(pf):.1f}ms")
    print(f"  Backbone:     mean={statistics.mean(bb):.1f}ms  std={statistics.stdev(bb):.1f}ms  min={min(bb):.1f}ms  max={max(bb):.1f}ms")
    print(f"  Acoustic:     mean={statistics.mean(ac):.1f}ms  std={statistics.stdev(ac):.1f}ms  min={min(ac):.1f}ms  max={max(ac):.1f}ms")
    print(f"Frames:         {results[0]['n_frames']}")
    print(f"Correctness:    all trials match={all_match}  warmup matches={warmup_matches}")
    print(f"SHA256:         {results[0]['sha256']}")


if __name__ == "__main__":
    main()
