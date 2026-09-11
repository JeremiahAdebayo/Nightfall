"""Smoke tests for the core algorithm components (CPU, no data needed)."""

import torch

from nightfall.config import (
    ALL_MVTEC_CATEGORIES,
    checkpoint_path,
    int8_checkpoint_path,
)
from nightfall.core.coreset import CoresetConfig, GreedyCoresetSampler
from nightfall.core.memory_bank import MemoryBank, MemoryBankConfig


def test_category_list_is_canonical():
    assert len(ALL_MVTEC_CATEGORIES) == 15
    assert len(set(ALL_MVTEC_CATEGORIES)) == 15


def test_checkpoint_path_convention():
    p = checkpoint_path("ckpts", "bottle")
    assert str(p).endswith("bottle_memory_bank.pt")


def test_int8_checkpoint_path_convention():
    p = int8_checkpoint_path("ckpts", "bottle")
    assert str(p).endswith("bottle_memory_bank_int8.pt")


def test_coreset_selects_requested_count():
    torch.manual_seed(0)
    features = torch.randn(1000, 32)
    sampler = GreedyCoresetSampler(CoresetConfig(sampling_ratio=0.1, projection_dim=None))
    selected = sampler.select(features)
    assert selected.shape == (100, 32)


def test_memory_bank_reweighting_flag_changes_image_score():
    torch.manual_seed(0)
    bank_tensor = torch.randn(50, 16)
    patches = torch.randn(28, 16)  # (B*H*W, D), as extract_patch_vectors returns it

    weighted = MemoryBank(MemoryBankConfig(use_reweighting=True))
    weighted.fit(bank_tensor)
    res_w = weighted.score(patches, (1, 4, 7), (224, 224))

    raw = MemoryBank(MemoryBankConfig(use_reweighting=False))
    raw.fit(bank_tensor)
    res_r = raw.score(patches, (1, 4, 7), (224, 224))

    assert not torch.allclose(res_w.image_score, res_r.image_score)
    # Pixel maps are raw distances in both paths -- they must match.
    assert torch.allclose(res_w.pixel_map, res_r.pixel_map, atol=1e-5)
