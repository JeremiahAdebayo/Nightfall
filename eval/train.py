"""
Train Nightfall's PatchCore across all MVTec AD categories.

Designed for Colab: the free tier can disconnect or recycle the runtime
without warning, so this script checkpoints each category's fitted
memory bank to disk *immediately* after fitting, and skips categories
that already have a saved checkpoint on restart. Point --output-dir at
a mounted Google Drive path (not /content) so checkpoints survive a
runtime recycle.

Usage (in a Colab cell):

    from google.colab import drive
    drive.mount('/content/drive')

    !python scripts/train_all_categories.py \
        --data-root /content/drive/MyDrive/mvtec_ad \
        --output-dir /content/drive/MyDrive/nightfall_checkpoints \
        --categories bottle cable capsule  # omit to train all 15

MVTec AD expected directory layout per category (standard download format):
    <data-root>/<category>/train/good/*.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from core.patchcore import PatchCore
from core.memory_bank import MemoryBank

ALL_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def checkpoint_path(output_dir: Path, category: str) -> Path:
    return output_dir / f"{category}_memory_bank.pt"


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


def ensure_mvtec_downloaded(data_root: Path, category: str) -> None:
    """
    Downloads and extracts one MVTec AD category using anomalib's MVTecAD
    datamodule, which handles the fetch + folder layout automatically --
    no manual download link required. A no-op if the category's data
    already exists on disk.

    We use anomalib here purely as a data-fetching utility (it already
    knows the correct MVTec folder structure and download source); the
    actual PatchCore algorithm is our own hand-rolled implementation in
    nightfall.core, not anomalib's.

    Note: the official MVTec download endpoint has occasionally returned
    404s (as of early 2026). If prepare_data() fails, fall back to the
    HuggingFace mirror at TheoM55/mvtec_all_objects_split, extracted into
    the same <data_root>/<category>/ layout by hand.
    """
    from anomalib.data import MVTecAD as AnomalibMVTecAD

    category_dir = data_root / category / "train" / "good"
    if category_dir.exists() and any(category_dir.glob("*.png")):
        return  # already downloaded

    datamodule = AnomalibMVTecAD(root=str(data_root), category=category)
    datamodule.prepare_data()


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

    ckpt_path = checkpoint_path(output_dir, category)
    torch.save(model.banks[category].bank, ckpt_path)

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
        if already_trained(args.output_dir, category):
            print(f"[skip] {category} already complete (checkpoint found)")
            bank = MemoryBank(model.bank_config)
            bank.fit(torch.load(checkpoint_path(args.output_dir, category)))
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
            # Continue to next category rather than aborting the whole run --
            # one bad category (missing data, corrupt image) shouldn't cost
            # you the other 14 on a long unattended run.
            continue

    completed = sum(1 for v in manifest.values() if v.get("status") == "complete")
    print(f"\nDone: {completed}/{len(args.categories)} categories trained successfully.")


if __name__ == "__main__":
    main()