"""
Memory bank storage and kNN-based anomaly scoring.

Scoring scheme (see DECISIONS.md):

Image-level anomaly score is NOT simply "distance from the single worst
patch to its nearest memory bank neighbor" (the naive max-min approach).
The PatchCore paper's re-weighting scheme instead asks, for each test
patch: how confident is the match to its nearest bank neighbor, relative
to the next-closest few candidates? We take the top-k nearest bank
neighbors' distances for a test patch and pass them through a softmax
(negated, so smaller distance = larger softmax mass). If the top-1
neighbor dominates that softmax (much closer than the runner-ups), the
match is confident and we trust the raw distance. If the top-k distances
are all similar (high-entropy softmax), the match is ambiguous -- the
patch sits roughly equidistant from several bank entries rather than
clearly close to one -- and we discount the raw distance accordingly.
Skipping this reweighting measurably increases false positives on
textures with legitimately high intra-class variation (e.g. wood grain,
carpet), where normal patches often have several similarly-plausible
bank matches rather than one clearly-best one.

Pixel-level anomaly map: each patch's raw (unweighted) nearest-neighbor
distance, reshaped to the feature map's spatial grid and upsampled to
image resolution via bilinear interpolation, then lightly Gaussian
blurred (per the paper) to smooth patch-grid quantization artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MemoryBankConfig:
    reweight_k: int = 3  # neighbors used for the reweighting factor
    gaussian_blur_sigma: float = 4.0
    gaussian_blur_kernel: int = 9


@dataclass
class ScoringResult:
    image_score: torch.Tensor  # (B,)
    pixel_map: torch.Tensor  # (B, H, W) upsampled to image resolution


class MemoryBank:
    """
    Stores a coreset of patch embeddings for one category and exposes
    kNN-based scoring against them.
    """

    def __init__(self, config: MemoryBankConfig | None = None):
        self.config = config or MemoryBankConfig()
        self.bank: torch.Tensor | None = None  # (M, D)

    def fit(self, coreset_features: torch.Tensor) -> None:
        self.bank = coreset_features.clone()

    def _gaussian_kernel(self, device: torch.device) -> torch.Tensor:
        k = self.config.gaussian_blur_kernel
        sigma = self.config.gaussian_blur_sigma
        coords = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel_2d = g[:, None] @ g[None, :]
        return kernel_2d.view(1, 1, k, k)

    @torch.no_grad()
    def score(
        self, patch_features: torch.Tensor, spatial_shape: tuple[int, int, int], image_hw: tuple[int, int]
    ) -> ScoringResult:
        """
        patch_features: (B*H*W, D) flattened patch embeddings for a batch
            (as returned by PatchFeatureExtractor.extract_patch_vectors).
        spatial_shape: (B, H, W) the feature map shape these patches came from.
        image_hw: (H_img, W_img) target resolution for the upsampled pixel map.
        """
        if self.bank is None:
            raise RuntimeError("MemoryBank.fit() must be called before scoring.")

        b, h, w = spatial_shape
        device = patch_features.device
        bank = self.bank.to(device)

        # Nearest neighbor distance from every test patch to the memory bank.
        dists = torch.cdist(patch_features, bank)  # (B*H*W, M)
        nn_dists = dists.min(dim=1).values  # (B*H*W,)

        # Reweighting factor, per the PatchCore paper: look at each test
        # patch's top-(reweight_k) nearest bank distances as a group.
        topk_test_to_bank = dists.topk(
            self.config.reweight_k, dim=1, largest=False
        ).values  # (B*H*W, k), sorted ascending; [:, 0] == nn_dists

        # softmax_weights[:, 0] is large (near 1) when the top-1 neighbor
        # clearly dominates -- a confident match. It's small (near 1/k)
        # when the k candidates are near-equidistant -- an ambiguous match.
        # We want confident matches to *preserve* the raw distance signal
        # (weight -> 1) and ambiguous matches to be discounted toward 0,
        # since an ambiguous match against a sparse/contested memory-bank
        # region is weaker evidence of anomaly than a confident one.
        softmax_weights = torch.softmax(-topk_test_to_bank, dim=1)  # (B*H*W, k)
        weight = softmax_weights[:, 0]  # confidence in the top-1 match

        weighted_dists = weight * nn_dists

        weighted_dists = weighted_dists.view(b, h * w)
        raw_dists = nn_dists.view(b, h, w)

        image_score = weighted_dists.max(dim=1).values  # (B,)

        pixel_map = raw_dists.unsqueeze(1)  # (B, 1, H, W)
        pixel_map = F.interpolate(
            pixel_map, size=image_hw, mode="bilinear", align_corners=False
        )
        kernel = self._gaussian_kernel(device)
        pad = self.config.gaussian_blur_kernel // 2
        pixel_map = F.conv2d(pixel_map, kernel, padding=pad)
        pixel_map = pixel_map.squeeze(1)  # (B, H_img, W_img)

        return ScoringResult(image_score=image_score, pixel_map=pixel_map)