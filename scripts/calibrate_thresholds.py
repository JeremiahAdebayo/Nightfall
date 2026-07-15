"""
Calibrate per-category anomaly thresholds using each category's own
training data (all "good"/normal images).

Why calibrate from TRAINING data, not test data: the memory bank is
built from training images, so scoring training images against their
own bank naturally produces a distribution of "normal" scores (with
some non-zero spread, since the memory bank is a coreset subsample, not
every training patch, so even a genuinely normal training image won't
score exactly 0). Setting the threshold using training scores avoids
peeking at test-set labels to pick a cutoff, which would be a form of
data leakage -- the threshold should be determined by what "normal"
looks like, not by tuning against the answers we're trying to detect.

Threshold rule: mean + 3*std of the training score distribution. This is
a standard, simple statistical convention (roughly 99.7% of a normal
distribution falls within 3 std of the mean) -- not claimed to be
optimal per-category, but a defensible, explainable default. A more
rigorous approach would use the actual test-set score distributions
(like the sorted-score inspection we did for hazelnut/metal_nut) to pick
a threshold that explicitly balances false positive vs false negative
rates for each category -- worth doing later per-category if the
mean+3std default proves too loose or too tight in practice.

Usage:
    !python scripts/calibrate_thresholds.py \
        --data-root {MVTEC_DIR} \
        --checkpoint-dir {DRIVE_ROOT}/checkpoints \
        --output {DRIVE_ROOT}/checkpoints/thresholds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core.patchcore import PatchCore
from core.memory_bank import MemoryBank
from scripts.train import ALL_MVTEC_CATEGORIES, checkpoint_path


def calibrate_category(model: PatchCore, category: str, data_root: Path) -> dict:
    train_dir = data_root / category / "train" / "good"
    train_paths = sorted(train_dir.glob("*.png"))

    train_images = model.preprocessor.batch(train_paths)
    scoring = model.predict(category, train_images)
    scores = scoring.image_score.detach().cpu().numpy()

    mean = float(scores.mean())
    std = float(scores.std())
    threshold = mean + 3 * std

    return {
        "mean": mean,
        "std": std,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "threshold": threshold,
        "num_train_images": len(train_paths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=ALL_MVTEC_CATEGORIES)
    args = parser.parse_args()

    model = PatchCore()
    thresholds = {}

    for category in args.categories:
        ckpt_path = checkpoint_path(args.checkpoint_dir, category)
        if not ckpt_path.exists():
            print(f"[{category}] SKIPPED -- no checkpoint found")
            continue

        bank = MemoryBank(model.bank_config)
        bank.fit(torch.load(ckpt_path))
        model.banks[category] = bank

        stats = calibrate_category(model, category, args.data_root)
        thresholds[category] = stats
        print(
            f"[{category}] mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
            f"threshold={stats['threshold']:.3f}  "
            f"(train range: {stats['min']:.3f} - {stats['max']:.3f})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(thresholds, indent=2))
    print(f"\nWrote {len(thresholds)} category thresholds to {args.output}")


if __name__ == "__main__":
    main()