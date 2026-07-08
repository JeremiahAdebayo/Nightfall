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
    <data-root>/<category>/test/<defect_type>/*.png
    <data-root>/<category>/ground_truth/<defect_type>/*_mask.png
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


def _download_via_huggingface_mirror(data_root: Path, category: str) -> None:
    """
    Fallback for when anomalib's MVTec download 404s (a documented,
    intermittent issue with the official endpoint as of early 2026).
    Pulls the same category from the community HuggingFace mirror
    TheoM55/mvtec_all_objects_split and writes it into the same on-disk
    layout ensure_mvtec_downloaded() expects:
        <category>/train/good/*.png
        <category>/test/<defect_type>/*.png
        <category>/ground_truth/<defect_type>/*_mask.png

    Only fetches what's actually missing on disk -- e.g. if train/ and
    test/ already exist from a prior run and only ground_truth/ is
    absent, this re-downloads the (small) test split to pull masks
    without re-saving train images that are already correctly present.
    """
    from datasets import load_dataset

    category_root = data_root / category
    train_dir = category_root / "train" / "good"
    test_dir = category_root / "test"
    gt_dir = category_root / "ground_truth"

    need_train = not (train_dir.exists() and any(train_dir.glob("*.png")))
    need_test_or_masks = not (
        test_dir.exists() and any(test_dir.rglob("*.png"))
        and gt_dir.exists() and any(gt_dir.rglob("*_mask.png"))
    )

    if need_train:
        train_dir.mkdir(parents=True, exist_ok=True)
        train_ds = load_dataset(
            "TheoM55/mvtec_all_objects_split", split=f"{category}.train"
        )
        for i, sample in enumerate(train_ds):
            sample["image_path"].save(train_dir / f"{i:03d}.png")

    if need_test_or_masks:
        test_ds = load_dataset(
            "TheoM55/mvtec_all_objects_split", split=f"{category}.test"
        )
        counts: dict[str, int] = {}
        for sample in test_ds:
            defect = sample["defect"]
            counts[defect] = counts.get(defect, 0) + 1
            idx = counts[defect]

            test_img_dir = test_dir / defect
            test_img_path = test_img_dir / f"{idx:03d}.png"
            if not test_img_path.exists():
                test_img_dir.mkdir(parents=True, exist_ok=True)
                sample["image_path"].save(test_img_path)

            # label == 1 marks a defective sample; "good" test images have
            # no mask (there's no defect to annotate), matching MVTec AD's
            # own convention of only providing ground_truth/ for non-good
            # classes.
            if sample.get("label") == 1 and sample.get("mask_path") is not None:
                mask_dir = gt_dir / defect
                mask_path = mask_dir / f"{idx:03d}_mask.png"
                if not mask_path.exists():
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    # MVTec's own naming convention suffixes mask filenames
                    # with "_mask" so they're distinguishable from the
                    # corresponding test image at the same numeric index.
                    sample["mask_path"].save(mask_path)


def _has_complete_data(data_root: Path, category: str) -> bool:
    """
    A category's on-disk data only counts as complete if train, test, AND
    ground_truth are all present. Checking train/ alone (the original,
    narrower check) let a category with training data but no masks get
    silently treated as "downloaded" -- fit() would succeed since it only
    needs train/good/, but eval later has no ground_truth/ to score
    against, and this check would never catch it on a re-run.
    """
    category_root = data_root / category
    train_dir = category_root / "train" / "good"
    test_dir = category_root / "test"
    gt_dir = category_root / "ground_truth"

    return (
        train_dir.exists() and any(train_dir.glob("*.png"))
        and test_dir.exists() and any(test_dir.rglob("*.png"))
        and gt_dir.exists() and any(gt_dir.rglob("*_mask.png"))
    )


def ensure_mvtec_downloaded(data_root: Path, category: str) -> None:
    """
    Downloads and extracts one MVTec AD category, preferring anomalib's
    MVTecAD datamodule (the documented, no-registration-link path), and
    falling back to a HuggingFace mirror if anomalib's download fails --
    which it does intermittently as of early 2026, per a known upstream
    issue with the official MVTec endpoint. A no-op if the category's
    train/test/ground_truth data already exists on disk, regardless of
    which path fetched it.

    We use anomalib here purely as a data-fetching utility (it already
    knows the correct MVTec folder structure and download source); the
    actual PatchCore algorithm is our own hand-rolled implementation in
    core/, not anomalib's.
    """
    if _has_complete_data(data_root, category):
        return  # train + test + ground_truth all already present

    category_dir = data_root / category / "train" / "good"

    try:
        from anomalib.data import MVTecAD as AnomalibMVTecAD

        datamodule = AnomalibMVTecAD(root=str(data_root), category=category)
        datamodule.prepare_data()

        # anomalib can fail silently past this point (e.g. write a
        # directory but no images) so verify the expected files actually
        # landed before declaring success -- don't just trust that
        # prepare_data() not raising means we have real data.
        if not _has_complete_data(data_root, category):
            raise RuntimeError(
                "anomalib prepare_data() completed but train/test/ground_truth "
                "are not all present"
            )

    except Exception as e:
        print(
            f"[{category}] anomalib download failed or incomplete ({e}); "
            f"falling back to HuggingFace mirror TheoM55/mvtec_all_objects_split"
        )
        _download_via_huggingface_mirror(data_root, category)


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
        # Data completeness (train/test/ground_truth) and training
        # completeness (a fitted, checkpointed memory bank) are two
        # different things -- a category can have a perfectly good
        # checkpoint from a prior run while still being missing masks
        # that were added to the download logic afterward. Always ensure
        # data is complete first, independent of whether we skip fitting.
        ensure_mvtec_downloaded(args.data_root, category)

        if already_trained(args.output_dir, category):
            print(f"[skip] {category} already trained (checkpoint found)")
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