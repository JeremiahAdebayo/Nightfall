"""
Run Phase 2 evaluation: loads each category's fitted memory bank
checkpoint, runs image AUROC / pixel AUROC / PRO against real MVTec test
data, and prints a results table.

Usage (Colab):
    !python scripts/run_eval.py \
        --data-root {MVTEC_DIR} \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.patchcore import PatchCore
from core.memory_bank import MemoryBank
from eval.harness import EvalHarness
from eval.dataloader import load_category_test_data

# Reused from train.py's convention rather than duplicated by hand, so
# the category list can't silently drift between training and eval.
from scripts.train import ALL_MVTEC_CATEGORIES, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--categories", nargs="+", default=ALL_MVTEC_CATEGORIES,
        help="Space-separated category names. Omit to evaluate all 15.",
    )
    args = parser.parse_args()

    model = PatchCore()
    harness = EvalHarness(model)

    for category in args.categories:
        ckpt_path = checkpoint_path(args.checkpoint_dir, category)
        if not ckpt_path.exists():
            print(f"[{category}] SKIPPED -- no checkpoint found at {ckpt_path}")
            continue

        bank = MemoryBank(model.bank_config)
        bank.fit(torch.load(ckpt_path))
        model.banks[category] = bank

        print(f"[{category}] loading test data...")
        try:
            test_data = load_category_test_data(
                args.data_root, category, model.preprocessor
            )
        except FileNotFoundError as e:
            print(f"[{category}] SKIPPED -- {e}")
            continue

        print(
            f"[{category}] evaluating "
            f"({len(test_data.image_labels)} test images, "
            f"bank size {model.memory_bank_size(category)})..."
        )
        result = harness.evaluate_category(test_data)
        print(
            f"[{category}] image_auroc={result.image_auroc:.4f}  "
            f"pixel_auroc={result.pixel_auroc:.4f}  "
            f"pro={result.pro_score:.4f}"
        )

    print("\n=== Summary ===")
    agg = harness.aggregate()
    if agg:
        print(
            f"Mean across {agg['num_categories']} categories: "
            f"image_auroc={agg['image_auroc']:.4f}  "
            f"pixel_auroc={agg['pixel_auroc']:.4f}  "
            f"pro={agg['pro_score']:.4f}"
        )
    else:
        print("No categories were successfully evaluated.")

    print("\n=== Comparison Table (markdown) ===")
    print(harness.comparison_table())

    # Original paper's headline mean numbers (Roth et al., 2021,
    # arXiv:2106.08265), corroborated across multiple independent
    # sources citing the same published figures. NOT per-category --
    # the original paper's per-category breakdown wasn't independently
    # verified against a clean primary source, so we deliberately don't
    # fabricate per-category paper numbers in the table above.
    print("\n=== Paper Reference (Roth et al., 2021) -- mean only ===")
    print(
        "Published mean: image_auroc=0.991  pixel_auroc=0.981  pro=0.935 "
        "(WideResNet50, 1% coreset, 224x224)"
    )
    print(
        "Note: per-category paper numbers are not shown above (marked '--') "
        "since a clean, independently-verified primary-source breakdown "
        "wasn't available at write time -- only the well-corroborated mean "
        "is reported here to avoid presenting unverified figures as fact."
    )


if __name__ == "__main__":
    main()