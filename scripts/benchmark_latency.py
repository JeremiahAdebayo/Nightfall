"""
Latency benchmark for Nightfall's feature extractor: PyTorch fp32 vs
ONNX fp32 vs ONNX INT8, all on CPU.

CPU-only is deliberate, not a limitation we're working around: Phase 3a's
investigation established that INT8 quantization's realistic execution
environment IS CPU (ONNX Runtime's CUDA provider doesn't implement
ConvInteger, the op INT8 conv relies on), and CPU is also the honest
proxy for actual edge hardware (Raspberry Pi, Jetson CPU cores,
microcontrollers) that this whole quantization effort is meant to serve.
Benchmarking GPU numbers here would measure something we can't actually
deploy this way, so we don't.

Reports p50/p95/p99 latency per single-image inference, plus throughput
(images/sec), across a fixed number of warm-up + measured runs.

Usage:
    !python scripts/benchmark_latency.py --onnx-dir {DRIVE_ROOT}/onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from core.feature_extractor import PatchFeatureExtractor


def percentile_stats(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "throughput_img_per_sec": float(1000.0 / arr.mean()),
    }


def benchmark_pytorch_fp32(input_size: int, num_warmup: int, num_runs: int) -> dict:
    model = PatchFeatureExtractor()
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy_input)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)

    return percentile_stats(latencies)


def benchmark_onnx(onnx_path: Path, input_size: int, num_warmup: int, num_runs: int) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    for _ in range(num_warmup):
        session.run([output_name], {input_name: dummy_input})

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run([output_name], {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    return percentile_stats(latencies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    args = parser.parse_args()

    fp32_onnx_path = args.onnx_dir / "feature_extractor.onnx"
    int8_onnx_path = args.onnx_dir / "feature_extractor_int8.onnx"

    print(f"Benchmarking on CPU, {args.num_runs} runs each (after {args.num_warmup} warmup runs)...\n")

    print("[1/3] PyTorch fp32...")
    pytorch_stats = benchmark_pytorch_fp32(args.input_size, args.num_warmup, args.num_runs)

    print("[2/3] ONNX fp32...")
    onnx_fp32_stats = benchmark_onnx(fp32_onnx_path, args.input_size, args.num_warmup, args.num_runs)

    print("[3/3] ONNX INT8...")
    onnx_int8_stats = benchmark_onnx(int8_onnx_path, args.input_size, args.num_warmup, args.num_runs)

    print("\n=== Results (single-image inference, CPU) ===")
    header = f"{'Model':<20} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'img/sec':>10}"
    print(header)
    print("-" * len(header))
    for name, stats in [
        ("PyTorch fp32", pytorch_stats),
        ("ONNX fp32", onnx_fp32_stats),
        ("ONNX INT8", onnx_int8_stats),
    ]:
        print(
            f"{name:<20} {stats['p50_ms']:>10.2f} {stats['p95_ms']:>10.2f} "
            f"{stats['p99_ms']:>10.2f} {stats['throughput_img_per_sec']:>10.1f}"
        )

    speedup_onnx = pytorch_stats["p50_ms"] / onnx_fp32_stats["p50_ms"]
    speedup_int8 = pytorch_stats["p50_ms"] / onnx_int8_stats["p50_ms"]
    print(f"\nONNX fp32 speedup over PyTorch: {speedup_onnx:.2f}x")
    print(f"ONNX INT8 speedup over PyTorch: {speedup_int8:.2f}x")


if __name__ == "__main__":
    main()