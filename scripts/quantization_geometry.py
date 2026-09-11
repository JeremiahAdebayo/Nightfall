"""
Quantization geometry analysis CLI: promotes Nightfall.ipynb cell 19 into
the repository so the analysis is runnable, reviewable, and re-runnable
without digging through a cleared-output notebook.

For each category, extracts the test set's patch features with BOTH the
native PyTorch fp32 extractor and the INT8 ONNX extractor, computes each
side's patch-to-bank distances, and reports how much of the kNN ranking
structure survives quantization (top-1 preservation, top-k overlap,
Spearman of top-k values, margin statistics -- see
nightfall.eval.quantization_geometry for what each metric means).

Bank choice (matches run_eval_int8.py's two modes):
  default  -- the EXISTING fp32 checkpoint bank (mismatched baseline; the
              geometry the original notebook cell measured).
  --bank-source int8 -- the refit bank saved by
              `run_eval_int8.py --refit-bank --save-refit`
              (<category>_memory_bank_int8.pt), i.e. the consistent
              pipeline's geometry.

Usage:
    python scripts/quantization_geometry.py \
        --data-root {MVTEC_DIR} \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints \
        --onnx-path {DRIVE_ROOT}/onnx/feature_extractor_int8.onnx \
        [--bank-source fp32|int8] [--categories bottle cable]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401 -- puts repo root on sys.path; see scripts/_bootstrap.py

from nightfall.config import (
    ALL_MVTEC_CATEGORIES,
    checkpoint_path,
    int8_checkpoint_path,
)
from nightfall.core.memory_bank import MemoryBank
from nightfall.core.onnx_feature_extractor import OnnxFeatureExtractor
from nightfall.core.patchcore import PatchCore
from nightfall.eval.dataloader import load_category_test_data
from nightfall.eval.quantization_geometry import geometry_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=ALL_MVTEC_CATEGORIES)
    parser.add_argument(
        "--bank-source",
        choices=("fp32", "int8"),
        default="fp32",
        help=(
            "Which bank both sides' distances are computed against: the "
            "fp32 checkpoint bank (default, mismatched baseline) or the "
            "INT8 refit bank saved by run_eval_int8.py --save-refit."
        ),
    )
    args = parser.parse_args()

    # PyTorch side supplies both the fp32 extractor and the preprocessing
    # config shared by both sides -- the same images must feed both
    # extractors or the comparison is meaningless.
    fp32_model = PatchCore()
    int8_extractor = OnnxFeatureExtractor(args.onnx_path)
    k = fp32_model.bank_config.reweight_k

    print(
        f"Quantization geometry analysis (k={k}, "
        f"bank-source={args.bank_source})"
    )

    results: dict[str, dict] = {}
    for category in args.categories:
        bank = MemoryBank(fp32_model.bank_config)
        if args.bank_source == "int8":
            int8_ckpt = int8_checkpoint_path(args.checkpoint_dir, category)
            if not int8_ckpt.exists():
                print(
                    f"[{category}] SKIPPED -- no refit bank at {int8_ckpt}; "
                    f"run scripts/run_eval_int8.py --refit-bank --save-refit "
                    f"first."
                )
                continue
            bank.fit(torch.load(int8_ckpt, weights_only=True))
        else:
            fp32_ckpt = checkpoint_path(args.checkpoint_dir, category)
            if not fp32_ckpt.exists():
                print(f"[{category}] SKIPPED -- no checkpoint found")
                continue
            bank.fit(torch.load(fp32_ckpt, weights_only=True))

        try:
            test_data = load_category_test_data(
                args.data_root, category, fp32_model.preprocessor
            )
        except FileNotFoundError as e:
            print(f"[{category}] SKIPPED -- {e}")
            continue

        vectors_fp32, _ = fp32_model.extractor.extract_patch_vectors(
            test_data.images
        )
        vectors_int8, _ = int8_extractor.extract_patch_vectors(test_data.images)

        dists_fp32 = torch.cdist(vectors_fp32, bank.bank)
        dists_int8 = torch.cdist(vectors_int8, bank.bank)

        metrics = geometry_metrics(dists_fp32, dists_int8, k=k)
        results[category] = metrics
        print(
            f"[{category}] top1={100 * metrics['top1_preservation']:.2f}%  "
            f"top{k}_overlap={100 * metrics['topk_overlap']:.2f}%  "
            f"spearman={metrics['spearman_corr']:.4f}  "
            f"margin_fp32={metrics['margin_mean_fp32']:.6f}  "
            f"margin_int8={metrics['margin_mean_int8']:.6f}  "
            f"margin_change={metrics['mean_margin_change']:.6f}"
        )

    if results:
        n = len(results)
        print(f"\n=== Summary ({n} categories, bank-source={args.bank_source}) ===")
        for key in ("top1_preservation", "topk_overlap"):
            mean = sum(r[key] for r in results.values()) / n
            print(f"mean {key}: {100 * mean:.2f}%")
        mean_corr = sum(r["spearman_corr"] for r in results.values()) / n
        print(f"mean spearman_corr: {mean_corr:.4f}")
        mean_change = sum(r["mean_margin_change"] for r in results.values()) / n
        print(f"mean margin_change: {mean_change:.6f}")


if __name__ == "__main__":
    main()
