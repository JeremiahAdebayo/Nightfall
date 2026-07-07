"""
Top-level PatchCore model: ties feature extraction, coreset selection,
and memory-bank scoring together, with per-category memory banks.

Multi-category design: rather than one giant memory bank across all
categories (which would let a bottle-shaped "normal" patch mask a
genuinely anomalous cable patch, since kNN scoring has no notion of
category boundaries), we keep one MemoryBank per category and route
fit/score calls by an explicit category key. This is more memory than
a shared bank but is the only correct approach given kNN scoring:
category identity is not something the algorithm can infer from patch
features alone, and mixing categories directly degrades scoring
precision within each one.
"""

from __future__ import annotations

import torch

from core.coreset import CoresetConfig, GreedyCoresetSampler
from core.feature_extractor import ExtractorConfig, PatchFeatureExtractor
from core.memory_bank import MemoryBank, MemoryBankConfig, ScoringResult
from core.preprocessing import ImagePreprocessor, PreprocessConfig


class PatchCore:
    def __init__(
        self,
        extractor_config: ExtractorConfig | None = None,
        coreset_config: CoresetConfig | None = None,
        bank_config: MemoryBankConfig | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ):
        self.extractor = PatchFeatureExtractor(extractor_config)
        self.coreset_sampler = GreedyCoresetSampler(coreset_config)
        self.bank_config = bank_config or MemoryBankConfig()
        self.preprocessor = ImagePreprocessor(preprocess_config)
        self.banks: dict[str, MemoryBank] = {}

    @torch.no_grad()
    def fit(self, category: str, normal_images: torch.Tensor) -> None:
        """
        normal_images: (N, 3, H, W) batch of nominal (defect-free) training
            images for a single category, already preprocessed (see
            fit_from_paths for raw-image input).
        """
        patch_vectors, _shape = self.extractor.extract_patch_vectors(normal_images)
        coreset = self.coreset_sampler.select(patch_vectors)

        bank = MemoryBank(self.bank_config)
        bank.fit(coreset)
        self.banks[category] = bank

    def fit_from_paths(self, category: str, image_paths: list) -> None:
        """
        Convenience entry point: preprocesses raw image files (paths, PIL
        Images) before delegating to `fit`. Prefer this over calling
        ImagePreprocessor manually so training and inference always share
        the exact same preprocessing config.
        """
        images = self.preprocessor.batch(image_paths)
        self.fit(category, images)

    @torch.no_grad()
    def predict(self, category: str, images: torch.Tensor) -> ScoringResult:
        """
        images: (N, 3, H, W) already preprocessed (see predict_from_paths
        for raw-image input).
        """
        if category not in self.banks:
            raise KeyError(
                f"No memory bank fitted for category '{category}'. "
                f"Known categories: {list(self.banks.keys())}"
            )

        patch_vectors, (b, h, w) = self.extractor.extract_patch_vectors(images)
        image_hw = images.shape[-2:]
        return self.banks[category].score(
            patch_vectors, spatial_shape=(b, h, w), image_hw=tuple(image_hw)
        )

    def predict_from_paths(self, category: str, image_paths: list) -> ScoringResult:
        """Convenience entry point: preprocesses raw image files before scoring."""
        images = self.preprocessor.batch(image_paths)
        return self.predict(category, images)

    @property
    def categories(self) -> list[str]:
        return list(self.banks.keys())

    def memory_bank_size(self, category: str) -> int:
        bank = self.banks.get(category)
        return 0 if bank is None or bank.bank is None else bank.bank.shape[0]