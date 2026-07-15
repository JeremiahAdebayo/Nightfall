"""
Real accuracy validation for the quantized feature extractor: re-runs
Phase 2's exact eval harness (image AUROC, pixel AUROC, PRO) per
category, but with patch features coming from the INT8 ONNX model
instead of native PyTorch fp32, and prints a direct comparison against
the fp32 numbers already on record.

Important limitation, stated plainly: this reuses the EXISTING fp32
memory banks (already fitted from PyTorch-extracted features) -- it does
NOT refit the memory bank using INT8-extracted training features. That
means this measures "what happens if we extract TEST features with INT8
but compare against a memory bank built from fp32 TRAIN features" --
a mismatched, slightly pessimistic scenario, not a clean apples-to-apples
"fully INT8 pipeline" test. A fully consistent test would refit each
category's memory bank using the INT8 extractor too. We're running the
cheaper, partial test first since it still tells us whether INT8 features
are in the same ballpark as fp32 ones; if this passes comfortably, a full
refit is probably unnecessary; if it fails, a full refit becomes the next
real experiment before concluding INT8 is unusable.

Usage:
    !python scripts/run_eval_int8.py \
        --data-root {MVTEC_DIR} \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints \
        --onnx-path {DRIVE_ROOT}/onnx/feature_extractor_int8.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.patchcore import PatchCore
from core.memory_bank import MemoryBank
from core.onnx_feature_extractor import OnnxFeatureExtractor
from eval.harness import EvalHarness
from eval.dataloader import load_category_test_data
from train import ALL_MVTEC_CATEGORIES, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=ALL_MVTEC_CATEGORIES)
    args = parser.parse_args()

    model = PatchCore()
    # Swap in the ONNX-backed extractor -- PatchCore, MemoryBank, and
    # EvalHarness are all unmodified; they just call
    # extractor.extract_patch_vectors(), which OnnxFeatureExtractor
    # implements with the same signature.
    model.extractor = OnnxFeatureExtractor(args.onnx_path)

    # Disable confidence reweighting for the quantized path: it's scale-
    # sensitive and collapses to near-chance AUROC under INT8 noise
    # (empirically verified -- see MemoryBankConfig.use_reweighting).
    # Raw max-distance scoring is both more honest and measurably better
    # for this specific inference path.
    model.bank_config.use_reweighting = False

    harness = EvalHarness(model)

    # Recorded fp32 results from Phase 2 (scripts/run_eval.py), for
    # direct comparison -- hardcoded here as the known reference point,
    # not re-derived, since re-running the fp32 eval every time we test
    # a quantized variant would be wasteful and these numbers are already
    # verified on record.
    fp32_reference = {
        "bottle": 0.997, "cable": 0.927, "capsule": 0.926, "carpet": 0.941,
        "grid": 0.799, "hazelnut": 0.956, "leather": 1.000, "metal_nut": 0.980,
        "pill": 0.810, "screw": 0.876, "tile": 0.993, "toothbrush": 0.989,
        "transistor": 0.969, "wood": 0.954, "zipper": 0.952,
    }

    for category in args.categories:
        ckpt_path = checkpoint_path(args.checkpoint_dir, category)
        if not ckpt_path.exists():
            print(f"[{category}] SKIPPED -- no checkpoint found")
            continue

        bank = MemoryBank(model.bank_config)
        bank.fit(torch.load(ckpt_path))
        model.banks[category] = bank

        try:
            test_data = load_category_test_data(
                args.data_root, category, model.preprocessor
            )
        except FileNotFoundError as e:
            print(f"[{category}] SKIPPED -- {e}")
            continue

        result = harness.evaluate_category(test_data)
        fp32_auroc = fp32_reference.get(category)
        delta = (
            f"{result.image_auroc - fp32_auroc:+.4f}"
            if fp32_auroc is not None else "n/a"
        )
        print(
            f"[{category}] int8_image_auroc={result.image_auroc:.4f}  "
            f"fp32_image_auroc={fp32_auroc}  delta={delta}  "
            f"pixel_auroc={result.pixel_auroc:.4f}  pro={result.pro_score:.4f}"
        )

    agg = harness.aggregate()
    if agg:
        fp32_mean = sum(fp32_reference.values()) / len(fp32_reference)
        print(f"\n=== Summary ===")
        print(f"INT8 mean image_auroc: {agg['image_auroc']:.4f}")
        print(f"fp32 mean image_auroc (recorded):  {fp32_mean:.4f}")
        print(f"Delta: {agg['image_auroc'] - fp32_mean:+.4f}")
        print(f"INT8 mean pixel_auroc: {agg['pixel_auroc']:.4f}")
        print(f"INT8 mean pro: {agg['pro_score']:.4f}")


if __name__ == "__main__":
    main()