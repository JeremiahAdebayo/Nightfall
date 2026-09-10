"""
Export PatchFeatureExtractor to ONNX, then verify numerical correctness
against the original PyTorch model on the same input.

Verification is not optional here: PatchFeatureExtractor.forward() reads
patch features back out of self._features, a plain Python dict populated
as a side effect of forward hooks on layer2/layer3. ONNX's tracer records
tensor operations executed during a forward pass, but whether it reliably
captures a hook-populated dict read as a genuine data dependency in the
exported graph is a known sharp edge, not something to assume works
without checking. If verification fails or is close-but-not-exact, that's
real information about whether this export path is trustworthy -- not
something to paper over.

Usage:
    python scripts/export_onnx.py --output-dir {DRIVE_ROOT}/onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401 -- puts repo root on sys.path; see scripts/_bootstrap.py

from nightfall.core.feature_extractor import PatchFeatureExtractor


def export_and_verify(output_dir: Path, input_size: int = 224) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "feature_extractor.onnx"

    model = PatchFeatureExtractor()
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Get the real PyTorch output first, on this exact input, so we have
    # a ground truth to compare the exported graph against.
    with torch.no_grad():
        torch_output = model(dummy_input)

    print(f"PyTorch output shape: {torch_output.shape}")

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["image"],
        output_names=["patch_features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "patch_features": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Export completed, file size: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Verification: load the exported graph and run it via onnxruntime,
    # compare against the PyTorch output on the SAME input tensor.
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(
        ["patch_features"], {"image": dummy_input.numpy()}
    )[0]

    torch_output_np = torch_output.numpy()

    if onnx_output.shape != torch_output_np.shape:
        print(
            f"[FAIL] Shape mismatch: ONNX {onnx_output.shape} vs "
            f"PyTorch {torch_output_np.shape}. This strongly suggests the "
            f"hook-based feature capture did NOT export correctly -- the "
            f"graph is structurally wrong, not just numerically imprecise."
        )
        return False

    max_abs_diff = np.abs(onnx_output - torch_output_np).max()
    mean_abs_diff = np.abs(onnx_output - torch_output_np).mean()
    print(f"Max absolute difference: {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")

    # A small numerical tolerance is expected and fine (different backend
    # kernels, floating point op ordering) -- this is NOT the same
    # tolerance question as the reweighting bug we caught earlier, since
    # here we're checking backend equivalence, not algorithm correctness.
    # A large or NaN difference means the export is structurally broken,
    # most likely due to the hook-based feature capture not being traced
    # as a real data dependency.
    tolerance = 1e-3
    if max_abs_diff > tolerance:
        print(
            f"[FAIL] Max difference {max_abs_diff:.6e} exceeds tolerance "
            f"{tolerance:.0e}. Do NOT trust this ONNX export -- the hook-based "
            f"forward() likely did not export correctly. Consider rewriting "
            f"forward() to return the fused feature map directly without "
            f"relying on hook-populated instance state, then re-export."
        )
        return False

    print(f"[PASS] ONNX export verified within tolerance ({tolerance:.0e}).")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    success = export_and_verify(args.output_dir, args.input_size)
    if not success:
        raise SystemExit(
            "ONNX export failed verification -- see output above for details. "
            "Do not proceed to quantization with an unverified export."
        )

if __name__ == "__main__":
    main()
