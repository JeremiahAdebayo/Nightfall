"""
Loads MVTec AD test-set images and ground-truth masks into the
CategoryTestData shape EvalHarness expects.

Directory convention (see train_all_categories.py's docstring):
    <data_root>/<category>/test/<defect_type>/*.png
    <data_root>/<category>/ground_truth/<defect_type>/*_mask.png

"good" test images have no ground-truth mask (there's no defect to
annotate) -- we fill in an all-zero mask for those rather than skipping
them, since pixel AUROC and PRO both need every pixel across the full
test set, including the true-negative pixels that only "good" images
contribute.

Mask/image pairing is by matching numeric index within a defect-type
folder (e.g. test/broken_large/001.png <-> ground_truth/broken_large/001_mask.png),
per the same convention train_all_categories.py's HuggingFace fallback
writes. This was spot-checked manually against several real image/mask
pairs before relying on it here (see project notes) -- the pairing is
NOT verified programmatically at load time, so a future change to
either side's naming convention could silently break this; a mismatch
would corrupt every downstream metric without raising an error, since
shapes would still line up even if content doesn't correspond.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.preprocessing import ImagePreprocessor
from eval.harness import CategoryTestData


def load_category_test_data(
    data_root: Path,
    category: str,
    preprocessor: ImagePreprocessor,
) -> CategoryTestData:
    """
    Walks <data_root>/<category>/test/ and .../ground_truth/, builds a
    CategoryTestData with preprocessed images, image-level labels
    (0=good, 1=defective), and pixel-level binary masks matched to the
    preprocessor's output resolution (crop_size x crop_size).
    """
    category_root = data_root / category
    test_root = category_root / "test"
    gt_root = category_root / "ground_truth"

    if not test_root.exists():
        raise FileNotFoundError(f"No test/ directory found at {test_root}")

    crop_size = preprocessor.config.crop_size

    image_paths: list[Path] = []
    image_labels: list[int] = []
    mask_arrays: list[np.ndarray] = []

    for defect_dir in sorted(test_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        is_good = defect_type == "good"

        for img_path in sorted(defect_dir.glob("*.png")):
            image_paths.append(img_path)
            image_labels.append(0 if is_good else 1)

            if is_good:
                # No defect to annotate -- an all-zero mask correctly
                # contributes true-negative pixels to pixel AUROC/PRO
                # without needing a real ground_truth/good/ folder to
                # exist (MVTec's own convention doesn't provide one).
                mask_arrays.append(np.zeros((crop_size, crop_size), dtype=np.uint8))
                continue

            # Match by numeric index within this defect type, per the
            # convention: 001.png <-> 001_mask.png.
            idx = img_path.stem  # e.g. "001"
            mask_path = gt_root / defect_type / f"{idx}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Expected mask at {mask_path} for test image {img_path}, "
                    f"but it doesn't exist. Pairing assumption may be broken "
                    f"for this category/defect_type -- do not silently skip, "
                    f"since that would misalign image_labels and pixel_labels "
                    f"arrays with the images tensor."
                )

            mask_img = Image.open(mask_path).convert("L")
            # Resize mask to match the preprocessed image's crop_size --
            # nearest-neighbor, not bilinear, since this is a binary mask
            # and interpolating would introduce fractional "defect-ish"
            # pixel values that don't correspond to a real annotation.
            mask_img = mask_img.resize((crop_size, crop_size), Image.NEAREST)
            mask_arr = (np.array(mask_img) > 127).astype(np.uint8)
            mask_arrays.append(mask_arr)

    if not image_paths:
        raise FileNotFoundError(f"No test images found under {test_root}")

    images = preprocessor.batch(image_paths)
    labels = np.array(image_labels, dtype=np.int64)
    masks = np.stack(mask_arrays, axis=0)

    return CategoryTestData(
        category=category,
        images=images,
        image_labels=labels,
        pixel_labels=masks,
    )