"""
Tests for the notebook-cell-19 metric core, on synthetic distance
matrices -- no model, dataset, or ONNX runtime involved.
"""

import torch

from nightfall.eval.quantization_geometry import geometry_metrics


def test_identical_distances_preserve_geometry_perfectly():
    torch.manual_seed(0)
    dists = torch.rand(64, 20)
    m = geometry_metrics(dists, dists.clone(), k=3)

    assert m["top1_preservation"] == 1.0
    assert m["topk_overlap"] == 1.0
    assert abs(m["spearman_corr"] - 1.0) < 1e-6
    assert m["mean_margin_change"] == 0.0


def test_swapped_runner_up_flips_top1_and_margin_but_not_spearman():
    """
    One patch's top-2 candidates swap order between fp32 and INT8: the
    nearest-neighbour identity flips (top-1 preservation 7/8) and the
    margin changes, but each side's top-k values stay monotonic, so the
    per-patch Spearman correlation is unaffected.
    """
    # 8 patches x 5 bank entries; rows 2..7 are identical on both sides.
    fp32 = torch.tensor(
        [
            [1.00, 1.10, 2.0, 3.0, 4.0],
            [1.00, 1.10, 2.0, 3.0, 4.0],
        ]
        + [[5.0, 6.0, 7.0, 8.0, 9.0]] * 6
    )
    int8 = fp32.clone()
    int8[0] = torch.tensor([1.20, 1.00, 3.5, 0.8, 4.0])  # top-1 flips 0 -> 3,
    # and the old runner-up (bank entry 2) falls out of the top-k entirely
    int8[1, 1] = 1.05  # top-2 value drifts; identity set unchanged

    m = geometry_metrics(fp32, int8, k=3)

    assert m["top1_preservation"] == 7 / 8
    # only patch 0 lost one candidate; isclose, not ==, because the mean of
    # per-patch overlaps accumulates float error past exact equality
    assert abs(m["topk_overlap"] - 23 / 24) < 1e-9
    assert m["spearman_corr"] == 1.0
    assert m["mean_margin_change"] > 0.0


def test_rejects_mismatched_shapes_and_bad_k():
    d = torch.rand(4, 10)
    try:
        geometry_metrics(d, torch.rand(4, 9), k=3)
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError as e:
        assert "shapes differ" in str(e)

    try:
        geometry_metrics(d, d.clone(), k=1)
        raise AssertionError("expected ValueError for k < 2")
    except ValueError as e:
        assert "k must be >= 2" in str(e)
