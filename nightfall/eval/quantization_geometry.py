"""
Quantization geometry analysis for the INT8 feature extractor.

Promoted from Nightfall.ipynb cell 19, where it lived only as a cleared-
output inline cell. This module holds the pure metric core; the CLI that
drives it against real checkpoints is scripts/quantization_geometry.py.

Why this analysis exists: aggregate AUROC deltas tell you *whether* INT8
scoring degrades, but not *why* or *how badly the ranking structure is
damaged*. These metrics compare, patch by patch, the k nearest-neighbour
geometry fp32 features produce against the one INT8 features produce over
the same bank:

- top-1 preservation -- how often the nearest bank entry is the SAME entry
  under both feature spaces. Nearest-neighbour identity, not distance
  magnitude, is what anomaly scoring actually consumes.
- top-k overlap -- how much of the k-candidate set survives. The
  reweighting factor reads exactly these k distances, so this is the
  quantity whose damage explains the reweighting collapse under INT8.
- Spearman correlation of the top-k distance values -- do the *relative*
  distances among candidates survive quantization.
- margin (top-2 minus top-1 distance) statistics -- how much separation
  there is between the best and runner-up match. Quantization noise that
  inflates margins uniformly is benign; noise that erases the margin
  makes the softmax confidence factor meaningless (weight -> 1/k), which
  is exactly the reweighting-collapse mechanism seen in the notebook.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def geometry_metrics(
    dists_fp32: torch.Tensor,
    dists_int8: torch.Tensor,
    k: int = 3,
) -> dict:
    """
    Compare the kNN geometry of two distance matrices over the same bank.

    dists_fp32, dists_int8: (P, M) patch-to-bank distance matrices for the
        SAME patches against the SAME bank, computed from fp32-extracted
        and INT8-extracted patch features respectively.
    k: neighbourhood size -- keep this equal to MemoryBankConfig.reweight_k
        so the analysis measures exactly the structure the reweighting
        factor consumes.

    Returns a plain dict of floats (see module docstring for semantics):
        top1_preservation, topk_overlap, spearman_corr,
        margin_mean_fp32, margin_mean_int8,
        margin_std_fp32, margin_std_int8, mean_margin_change
    """
    if dists_fp32.shape != dists_int8.shape:
        raise ValueError(
            f"Distance matrix shapes differ: fp32 {tuple(dists_fp32.shape)} "
            f"vs int8 {tuple(dists_int8.shape)} -- the two sides must be "
            f"computed over the same patches and the same bank."
        )
    if k < 2:
        raise ValueError("k must be >= 2: the margin metric needs a runner-up.")
    m = dists_fp32.shape[1]
    if k > m:
        raise ValueError(f"k={k} exceeds the bank size ({m} entries).")

    topk_fp32_vals, topk_fp32_idx = dists_fp32.topk(k, largest=False)
    topk_int8_vals, topk_int8_idx = dists_int8.topk(k, largest=False)

    top1_preservation = (
        (topk_fp32_idx[:, 0] == topk_int8_idx[:, 0]).float().mean().item()
    )

    overlaps = [
        len(set(a.tolist()) & set(b.tolist())) / k
        for a, b in zip(topk_fp32_idx, topk_int8_idx)
    ]
    topk_overlap = float(np.mean(overlaps))

    # Per-patch Spearman between the two sides' top-k distance VALUES,
    # sorted ascending on each side. A constant row (degenerate input,
    # e.g. a synthetic test) yields nan -- skip it the way the original
    # notebook cell did rather than failing the whole analysis.
    corrs = []
    for fp, q in zip(topk_fp32_vals.numpy(), topk_int8_vals.numpy()):
        res = spearmanr(fp, q)
        c = res.statistic if hasattr(res, "statistic") else res[0]
        if not np.isnan(c):
            corrs.append(float(c))
    spearman_corr = float(np.mean(corrs)) if corrs else float("nan")

    margin_fp32 = (topk_fp32_vals[:, 1] - topk_fp32_vals[:, 0]).numpy()
    margin_int8 = (topk_int8_vals[:, 1] - topk_int8_vals[:, 0]).numpy()

    return {
        "top1_preservation": top1_preservation,
        "topk_overlap": topk_overlap,
        "spearman_corr": spearman_corr,
        "margin_mean_fp32": float(margin_fp32.mean()),
        "margin_mean_int8": float(margin_int8.mean()),
        "margin_std_fp32": float(margin_fp32.std()),
        "margin_std_int8": float(margin_int8.std()),
        "mean_margin_change": float(np.mean(np.abs(margin_fp32 - margin_int8))),
    }
