"""
Metric behaviour tests on synthetic data -- the metric code is pure
numpy/scipy/sklearn, so these run without torchvision or a dataset.
"""

import numpy as np
import pytest

from nightfall.eval.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_score,
)


def test_image_auroc_perfect_and_chance():
    labels = np.array([0, 0, 1, 1])
    assert compute_image_auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    # Inverted ranking is the exact complement of a perfect separation.
    assert compute_image_auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0


def test_image_auroc_requires_both_classes():
    with pytest.raises(ValueError):
        compute_image_auroc(np.array([0.1, 0.2]), np.array([0, 0]))


def test_pixel_auroc_perfect_localization():
    labels = np.zeros((1, 8, 8), dtype=np.uint8)
    labels[0, 2:5, 2:5] = 1
    scores = np.where(labels == 1, 1.0, 0.0)
    assert compute_pixel_auroc(scores, labels) == 1.0


def test_pro_penalizes_partial_region_coverage():
    """
    PRO averages recall per connected region, so a model that localizes the
    top half of a defect and misses the rest scores far below one that
    covers it -- a gap plain pixel AUROC hides.

    Scores use a spread background (not a flat one): with only two distinct
    score values in the map, the PRO-vs-FPR curve has no FPR width at all
    and the trapezoid integrates over zero area. Real score maps are
    continuous, so this degenerate case only arises from synthetic input.
    """
    rng = np.random.default_rng(0)

    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[2:6, 2:6] = 1  # one 4x4 defect region
    background = rng.uniform(0, 1, (8, 8))

    covered = np.where(labels == 1, 2.0, background)[None].astype(np.float32)
    half_missed = covered.copy()
    # Bottom half of the region scores *below* the normal background.
    half_missed[0, 4:6, 2:6] = (background[4:6, 2:6] * 0.01).astype(np.float32)

    assert compute_pro_score(covered, labels[None]) > 0.9
    assert compute_pro_score(half_missed, labels[None]) < 0.7


def test_pro_weights_regions_equally_regardless_of_size():
    """
    Two defects of different sizes contribute equally to PRO, so failing one
    of two regions costs roughly as much as failing either one -- the point
    of averaging over regions instead of pixels.
    """
    rng = np.random.default_rng(1)

    labels = np.zeros((8, 16), dtype=np.uint8)
    labels[1:3, 1:4] = 1  # 6 px region
    labels[5:7, 10:15] = 1  # 10 px region
    background = rng.uniform(0, 1, (8, 16))

    scores = np.where(labels == 1, 3.0, background)[None].astype(np.float32)
    missed_small = scores.copy()
    missed_small[0, 1:3, 1:4] = (background[1:3, 1:4] * 0.01).astype(np.float32)
    missed_large = scores.copy()
    missed_large[0, 5:7, 10:15] = (background[5:7, 10:15] * 0.01).astype(np.float32)

    full = compute_pro_score(scores, labels[None])
    drop_small = full - compute_pro_score(missed_small, labels[None])
    drop_large = full - compute_pro_score(missed_large, labels[None])

    # Both drops are large and of the same order, even though the missed
    # regions differ in size by ~1.7x -- pixel-AUROC-style weighting would
    # make the large region dominate.
    assert drop_small > 0.3 and drop_large > 0.3
