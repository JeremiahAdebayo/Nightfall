"""
INT8 post-training quantization of Nightfall's exported feature
extractor, using ONNX Runtime's dynamic quantization (no calibration
dataset required -- quantizes weights statically, activations
dynamically at inference time based on observed ranges).

Why dynamic over static quantization here: static quantization needs a
representative calibration dataset to pre-compute activation ranges,
which gives slightly better accuracy/latency but adds a real pipeline
step (running calibration images through the model first). Dynamic
quantization is a reasonable, simpler first cut for establishing the
size/accuracy tradeoff -- if the accuracy delta (checked below) turns
out to be unacceptable, static quantization with real MVTec calibration
images is the natural next experiment, not something to reach for
by default.

This script does NOT just quantize and declare victory -- it re-runs
the same PyTorch-vs-ONNX verification from export_onnx.py, but now
comparing INT8 ONNX output against the original fp32 PyTorch output,
since quantization is expected to introduce real (not just floating-point
rounding) numerical differences, and we need to know how large those
differences actually are before trusting this for anomaly scoring.

Usage:
    python scripts/quantize_onnx.py --onnx-dir {DRIVE_ROOT}/onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401 -- puts repo root on sys.path; see scripts/_bootstrap.py

from nightfall.core.feature_extractor import PatchFeatureExtractor


def quantize_and_verify(onnx_dir: Path, input_size: int = 224) -> dict:
    fp32_path = onnx_dir / "feature_extractor.onnx"
    int8_path = onnx_dir / "feature_extractor_int8.onnx"

    if not fp32_path.exists():
        raise FileNotFoundError(
            f"No fp32 ONNX model found at {fp32_path} -- run "
            f"scripts/export_onnx.py first."
        )

    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing {fp32_path} -> {int8_path}...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    # Size comparison -- include external-data companion files, since
    # (as we confirmed with the fp32 export) the .onnx file alone is not
    # the real size; large weight tensors live in a sidecar .data file.
    def total_size_mb(onnx_path: Path) -> float:
        total_bytes = onnx_path.stat().st_size
        data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
        if data_path.exists():
            total_bytes += data_path.stat().st_size
        return total_bytes / 1e6

    fp32_size = total_size_mb(fp32_path)
    int8_size = total_size_mb(int8_path)
    print(f"fp32 total size (graph + external data): {fp32_size:.1f} MB")
    print(f"int8 total size (graph + external data): {int8_size:.1f} MB")
    print(f"Size reduction: {fp32_size / int8_size:.2f}x")

    # Accuracy verification: compare INT8 ONNX output against the
    # ORIGINAL PyTorch fp32 model, not against the fp32 ONNX export --
    # we want to know the real end-to-end drift from "what the model
    # originally computed" to "what we'll actually run at inference",
    # not just the quantization step in isolation.
    import onnxruntime as ort

    model = PatchFeatureExtractor()
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    session = ort.InferenceSession(str(int8_path))
    int8_output = session.run(
        ["patch_features"], {"image": dummy_input.numpy()}
    )[0]

    if int8_output.shape != torch_output.shape:
        raise RuntimeError(
            f"Shape mismatch after quantization: INT8 {int8_output.shape} vs "
            f"PyTorch {torch_output.shape}. Quantization broke the graph "
            f"structure -- do not trust this model."
        )

    max_abs_diff = float(np.abs(int8_output - torch_output).max())
    mean_abs_diff = float(np.abs(int8_output - torch_output).mean())
    # Relative difference matters more than absolute here, since these
    # are patch feature activations, not normalized probabilities --
    # their natural scale varies across channels, so a fixed absolute
    # tolerance is less meaningful than for a bounded output.
    relative_diff = float(
        np.abs(int8_output - torch_output).mean()
        / (np.abs(torch_output).mean() + 1e-8)
    )

    print(f"Max absolute difference vs original PyTorch fp32: {max_abs_diff:.4f}")
    print(f"Mean absolute difference: {mean_abs_diff:.4f}")
    print(f"Mean relative difference: {relative_diff:.2%}")
    print(
        "\nNOTE: this is a single random input, not a real accuracy "
        "measurement. A meaningful accuracy check means re-running "
        "scripts/run_eval.py's image/pixel AUROC and PRO metrics with "
        "the memory bank fed INT8-extracted features instead of fp32 "
        "ones, and comparing against the Phase 2 numbers already on "
        "record. Size/latency numbers above are trustworthy on their "
        "own; accuracy impact is NOT confirmed until that comparison "
        "is run."
    )

    return {
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "size_reduction_factor": fp32_size / int8_size,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_relative_diff": relative_diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    results = quantize_and_verify(args.onnx_dir, args.input_size)
    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
