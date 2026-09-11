"""
Train Nightfall's PatchCore across all MVTec AD categories.

Thin CLI entry point: the training logic (per-category fit + checkpoint)
and dataset acquisition (nightfall/data/mvtec.py) live in the nightfall
package; this script only wires arguments to them.

Designed for Colab: the free tier can disconnect or recycle the runtime
without warning, so each category's memory bank is checkpointed to disk
immediately after fitting, and already-trained categories are skipped on
restart. Point --output-dir at a mounted Google Drive path (not /content)
so checkpoints survive a runtime recycle.

Usage:
    python scripts/train.py --data-root <mvtec_dir> --output-dir <ckpt_dir> \
        [--categories bottle cable]   # omit to train all 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import _bootstrap  # noqa: F401 -- puts repo root on sys.path; see scripts/_bootstrap.py

from nightfall.config import ALL_MVTEC_CATEGORIES, checkpoint_path
from nightfall.core.memory_bank import MemoryBank
from nightfall.core.patchcore import PatchCore
from nightfall.data.mvtec import ensure_mvtec_downloaded


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "training_manifest.json"


def load_manifest(output_dir: Path) -> dict:
    path = manifest_path(output_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_path(output_dir).write_text(json.dumps(manifest, indent=2))


def already_trained(output_dir: Path, category: str) -> bool:
    """
    A category counts as done only if BOTH the checkpoint file exists AND
    the manifest confirms it completed successfully -- guards against a
    partially-written checkpoint from a run that died mid-torch.save.
    """
    manifest = load_manifest(output_dir)
    return (
        checkpoint_path(output_dir, category).exists()
        and manifest.get(category, {}).get("status") == "complete"
    )


def train_category(
    model: PatchCore,
    category: str,
    data_root: Path,
    output_dir: Path,
) -> dict:
    ensure_mvtec_downloaded(data_root, category)

    train_dir = data_root / category / "train" / "good"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected training images at {train_dir} -- check --data-root "
            f"matches the standard MVTec AD download layout."
        )

    image_paths = sorted(train_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No .png images found in {train_dir}")

    start = time.time()
    model.fit_from_paths(category, image_paths)
    elapsed = time.time() - start

    torch.save(model.banks[category].bank, checkpoint_path(output_dir, category))

    return {
        "status": "complete",
        "num_train_images": len(image_paths),
        "memory_bank_size": model.memory_bank_size(category),
        "fit_time_seconds": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=ALL_MVTEC_CATEGORIES,
        help="Space-separated category names. Omit to train all 15.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.output_dir)

    print(f"Training {len(args.categories)} categories: {args.categories}")
    model = PatchCore()

    for category in args.categories:
        # Data completeness (train/test/ground_truth) and training
        # completeness (a fitted, checkpointed memory bank) are two
        # different things -- always ensure data is complete first,
        # independent of whether we skip fitting.
        ensure_mvtec_downloaded(args.data_root, category)

        if already_trained(args.output_dir, category):
            print(f"[skip] {category} already trained (checkpoint found)")
            bank = MemoryBank(model.bank_config)
            bank.fit(
                torch.load(
                    checkpoint_path(args.output_dir, category), weights_only=True
                )
            )
            model.banks[category] = bank
            continue

        print(f"[start] {category}")
        try:
            result = train_category(model, category, args.data_root, args.output_dir)
            manifest[category] = result
            save_manifest(args.output_dir, manifest)
            print(
                f"[done]  {category} -- "
                f"{result['num_train_images']} images, "
                f"bank size {result['memory_bank_size']}, "
                f"{result['fit_time_seconds']}s"
            )
        except Exception as e:
            manifest[category] = {"status": "failed", "error": str(e)}
            save_manifest(args.output_dir, manifest)
            print(f"[FAIL]  {category}: {e}")
            # One bad category (missing data, corrupt image) shouldn't cost
            # you the other 14 on a long unattended run.
            continue

    completed = sum(1 for v in manifest.values() if v.get("status") == "complete")
    print(f"\nDone: {completed}/{len(args.categories)} categories trained successfully.")


if __name__ == "__main__":
    main()
