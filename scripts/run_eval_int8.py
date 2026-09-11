"""
Real accuracy validation for the quantized feature extractor: re-runs
Phase 2's exact eval harness (image AUROC, pixel AUROC, PRO) per
category, but with patch features coming from the INT8 ONNX model
instead of native PyTorch fp32, and prints a direct comparison against
the fp32 numbers already on record.

Two modes:

Default (mismatched-bank baseline): reuses the EXISTING fp32 memory
banks (fitted from PyTorch-extracted features) and extracts only TEST
features with INT8. This measures the pessimistic scenario "INT8 test
features scored against a bank built from fp32 train features" -- the
cheaper first check, but NOT the fully-INT8 pipeline. The notebook's
refit investigation (cells 23-24) later showed the mismatch itself,
not quantization, caused most of the collapse: refitting the bank with
the same INT8 extractor restored hazelnut 0.486 -> 1.000 and
metal_nut 0.623 -> 0.998.

--refit-bank (consistent pipeline): refits each category's memory bank
from <data-root>/<category>/train/good using the SAME INT8 extractor
used for scoring, then evaluates. This is the real experiment -- the
train/test feature-space invariant in DECISION.md says the extractor
that builds a bank and the extractor that scores against it must be
the same. With --save-refit the refit bank is written to
<checkpoint-dir>/<category>_memory_bank_int8.pt (never overwriting the
fp32 bank) and later --refit-bank runs reload it automatically instead
of re-extracting the whole training set.

Usage:
    python scripts/run_eval_int8.py \
        --data-root {MVTEC_DIR} \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints \
        --onnx-path {DRIVE_ROOT}/onnx/feature_extractor_int8.onnx
        [--refit-bank] [--save-refit]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401 -- puts repo root on sys.path; see scripts/_bootstrap.py

from nightfall.core.patchcore import PatchCore
from nightfall.core.memory_bank import MemoryBank
from nightfall.core.onnx_feature_extractor import OnnxFeatureExtractor
from nightfall.eval.harness import EvalHarness
from nightfall.eval.dataloader import load_category_test_data
from nightfall.config import (
    ALL_MVTEC_CATEGORIES,
    checkpoint_path,
    int8_checkpoint_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=ALL_MVTEC_CATEGORIES)
    parser.add_argument(
        "--refit-bank",
        action="store_true",
        help=(
            "Refit each category's memory bank from train/good with the "
            "same INT8 extractor used for scoring (the consistent "
            "feature-space pipeline), instead of scoring INT8 features "
            "against the fp32-built banks."
        ),
    )
    parser.add_argument(
        "--save-refit",
        action="store_true",
        help=(
            "With --refit-bank, save each refit bank to "
            "<checkpoint-dir>/<category>_memory_bank_int8.pt so later "
            "runs reload it instead of re-extracting the training set."
        ),
    )
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
        if args.refit_bank:
            int8_ckpt = int8_checkpoint_path(args.checkpoint_dir, category)
            if int8_ckpt.exists():
                bank = MemoryBank(model.bank_config)
                bank.fit(torch.load(int8_ckpt, weights_only=True))
                model.banks[category] = bank
                print(f"[{category}] loaded previously refit INT8 bank")
            else:
                train_dir = args.data_root / category / "train" / "good"
                image_paths = (
                    sorted(train_dir.glob("*.png")) if train_dir.exists() else []
                )
                if not image_paths:
                    print(f"[{category}] SKIPPED -- no training images at {train_dir}")
                    continue
                print(
                    f"[{category}] refitting bank from {len(image_paths)} "
                    f"train images with the INT8 extractor..."
                )
                # fit_from_paths preprocesses and runs coreset selection
                # exactly as training did, but against the INT8 extractor
                # installed on the model -- the bank and the scorer now
                # share one feature space by construction.
                model.fit_from_paths(category, image_paths)
                if args.save_refit:
                    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model.banks[category].bank, int8_ckpt)
                    print(f"[{category}] saved refit bank to {int8_ckpt.name}")
        else:
            ckpt_path = checkpoint_path(args.checkpoint_dir, category)
            if not ckpt_path.exists():
                print(f"[{category}] SKIPPED -- no checkpoint found")
                continue

            bank = MemoryBank(model.bank_config)
            bank.fit(torch.load(ckpt_path, weights_only=True))
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
            f"pixel_auroc={result.pixel_auroc:.4f}  pro={result.pro_score:.4f}  "
            f"bank_size={model.memory_bank_size(category)}"
        )

    agg = harness.aggregate()
    if agg:
        fp32_mean = sum(fp32_reference.values()) / len(fp32_reference)
        print(f"\n=== Summary ===")
        mode = (
            "refit (consistent INT8 pipeline)"
            if args.refit_bank
            else "mismatched-bank baseline (fp32 banks, INT8 test features)"
        )
        print(f"Mode: {mode}")
        print(f"INT8 mean image_auroc: {agg['image_auroc']:.4f}")
        print(f"fp32 mean image_auroc (recorded):  {fp32_mean:.4f}")
        print(f"Delta: {agg['image_auroc'] - fp32_mean:+.4f}")
        print(f"INT8 mean pixel_auroc: {agg['pixel_auroc']:.4f}")
        print(f"INT8 mean pro: {agg['pro_score']:.4f}")


if __name__ == "__main__":
    main()
