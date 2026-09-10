"""
MVTec AD dataset acquisition: download per-category data via anomalib's
datamodule, with a HuggingFace mirror fallback, into the standard on-disk
layout:

    <data-root>/<category>/train/good/*.png
    <data-root>/<category>/test/<defect_type>/*.png
    <data-root>/<category>/ground_truth/<defect_type>/*_mask.png

Extracted from the original scripts/train.py so data fetching lives with
data (nightfall/data/) and the training CLI stays thin.
"""

from __future__ import annotations

from pathlib import Path

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


