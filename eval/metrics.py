"""
Evaluation metrics for anomaly detection.

Why PRO exists alongside AUROC (see DECISIONS.md):

Pixel-level AUROC is computed over every pixel in the dataset, and the
overwhelming majority of pixels in any image are background/normal --
even the entire image, for non-defective test samples. A model that
produces a diffuse, low-confidence anomaly map covering only 20% of a
true defect region can still score a very high pixel AUROC, because
AUROC doesn't care whether that 20% was spread across many small defects
or concentrated in one large one, and it doesn't penalize under-detection
of a specific region as long as ranking is preserved elsewhere.

PRO (Per-Region Overlap) instead operates per connected defect region:
for each ground-truth defect region, compute what fraction of that
region's pixels are flagged as anomalous at a given threshold, then
average that fraction *across regions*, not weighted by region size.
This means a model that reliably catches small defects but is sloppy on
large ones (or vice versa) gets penalized in a way plain pixel AUROC
would hide. The final PRO score integrates this per-region-recall curve
over false-positive rates from 0 to a cutoff (conventionally 0.3), as in
the original PatchCore/Bergmann et al. formulation, then normalizes the
area under that curve to [0, 1].
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def compute_image_auroc(image_scores: np.ndarray, image_labels: np.ndarray) -> float:
    """
    image_scores: (N,) anomaly scores, higher = more anomalous.
    image_labels: (N,) binary ground truth, 1 = defective, 0 = normal.
    """
    if len(np.unique(image_labels)) < 2:
        raise ValueError(
            "compute_image_auroc requires both normal and defective samples; "
            f"got labels {np.unique(image_labels).tolist()}"
        )
    return float(roc_auc_score(image_labels, image_scores))


def compute_pixel_auroc(pixel_scores: np.ndarray, pixel_labels: np.ndarray) -> float:
    """
    pixel_scores: (N, H, W) per-pixel anomaly scores.
    pixel_labels: (N, H, W) binary ground truth masks.
    Flattened internally -- this treats every pixel across every image as
    one large pool, per standard practice.
    """
    flat_scores = pixel_scores.reshape(-1)
    flat_labels = pixel_labels.reshape(-1)
    if len(np.unique(flat_labels)) < 2:
        raise ValueError(
            "compute_pixel_auroc requires both normal and anomalous pixels "
            "across the provided samples."
        )
    return float(roc_auc_score(flat_labels, flat_scores))


def compute_pro_score(
    pixel_scores: np.ndarray,
    pixel_labels: np.ndarray,
    max_fpr: float = 0.3,
    num_thresholds: int = 50,
) -> float:
    """
    pixel_scores: (N, H, W) per-pixel anomaly scores.
    pixel_labels: (N, H, W) binary ground truth masks (0/1).
    max_fpr: integrate the PRO-vs-FPR curve only up to this false-positive
        rate, per the standard PRO formulation -- beyond this point
        thresholds are so permissive that the metric stops being
        discriminative between models.
    num_thresholds: number of threshold points sampled between the min and
        max observed score to build the curve. More points = smoother
        integration at higher compute cost; 50 is a reasonable default
        for reporting, not for tight optimization loops.

    Returns the normalized area under the PRO-vs-FPR curve in [0, max_fpr],
    scaled to [0, 1] (i.e. divided by max_fpr) so it's comparable across
    different max_fpr choices and directly comparable to AUROC's [0, 1] scale.
    """
    thresholds = np.linspace(pixel_scores.min(), pixel_scores.max(), num_thresholds)

    # Precompute connected components of ground-truth defect regions per
    # image, once -- these don't depend on threshold.
    labeled_regions = []
    for labels_img in pixel_labels:
        labeled, num_regions = ndimage.label(labels_img)
        regions = [
            (labeled == region_id) for region_id in range(1, num_regions + 1)
        ]
        labeled_regions.append(regions)

    normal_mask_total = (pixel_labels == 0).sum()

    pro_values = []
    fpr_values = []

    for thresh in thresholds:
        predicted = pixel_scores >= thresh

        # Per-region recall, averaged across all regions in the dataset
        # (not weighted by region size -- a 5px scratch counts as much
        # as a 500px stain, which is the whole point of PRO).
        region_recalls = []
        for img_idx, regions in enumerate(labeled_regions):
            for region_mask in regions:
                region_size = region_mask.sum()
                if region_size == 0:
                    continue
                overlap = (predicted[img_idx] & region_mask).sum()
                region_recalls.append(overlap / region_size)

        pro = float(np.mean(region_recalls)) if region_recalls else 0.0

        # False positive rate over normal (label == 0) pixels only.
        false_positives = (predicted & (pixel_labels == 0)).sum()
        fpr = float(false_positives / normal_mask_total) if normal_mask_total > 0 else 0.0

        pro_values.append(pro)
        fpr_values.append(fpr)

    # Sort by FPR ascending for integration, restrict to [0, max_fpr].
    order = np.argsort(fpr_values)
    fpr_sorted = np.array(fpr_values)[order]
    pro_sorted = np.array(pro_values)[order]

    mask = fpr_sorted <= max_fpr
    if mask.sum() < 2:
        # Not enough points below max_fpr to integrate meaningfully --
        # this typically means max_fpr is too tight for the threshold
        # granularity used; widen num_thresholds or max_fpr.
        return 0.0

    fpr_clipped = fpr_sorted[mask]
    pro_clipped = pro_sorted[mask]

    area = np.trapz(pro_clipped, fpr_clipped)
    return float(area / max_fpr)