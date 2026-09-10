"""
Greedy coreset subsampling for PatchCore memory banks.

Why greedy k-center over random subsampling (see DECISIONS.md):

A full memory bank of every training patch embedding is enormous (millions
of vectors for a modest training set) and mostly redundant -- neighboring
pixels of a normal, textured surface produce near-duplicate patch features.
Random subsampling reduces size but has no guarantee about *coverage*: it
can easily drop an entire mode of "normal" appearance (e.g. one lighting
condition, one region of the object) if that mode is a minority of patches,
which then causes false positives at inference when a legitimately normal
patch has no nearby neighbor left in the bank.

Greedy k-center subsampling instead selects points that maximize the
minimum distance to the already-selected set at each step -- i.e. it
greedily picks the point that is currently *worst covered*. This gives a
formal guarantee: the maximum distance from any dropped point to its
nearest retained point is bounded by twice the optimal k-center radius.
In practice this means far better coverage of rare-but-normal patch types
at a given memory bank size, which is the entire point of subsampling in
the first place -- we're trying to keep coverage, not just save memory.

Implementation note: exact k-center is NP-hard; the standard greedy
approximation (Farthest Point Sampling) is what the PatchCore paper uses
and what we implement here. For memory banks with more than ~50k candidate
patches, we additionally use the paper's random projection trick (via
Johnson-Lindenstrauss) to cut the distance-computation cost, since greedy
FPS is O(n*k) in the *embedding dimension*, which dominates for
high-dimensional WideResNet features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CoresetConfig:
    sampling_ratio: float = 0.01  # fraction of patches to retain
    projection_dim: int | None = 128  # None disables JL projection
    seed: int = 0


class GreedyCoresetSampler:
    """
    Farthest Point Sampling (greedy k-center approximation) for selecting
    a representative subset of patch embeddings.
    """

    def __init__(self, config: CoresetConfig | None = None):
        self.config = config or CoresetConfig()

    def _maybe_project(self, features: torch.Tensor) -> torch.Tensor:
        """
        Random projection (Johnson-Lindenstrauss) to a lower dimension for
        the *distance computation only* -- selected indices are then used
        to index into the original, unprojected features. This preserves
        pairwise distances approximately while making the O(n*k*d) greedy
        selection loop cheaper when d (embedding dim) is large.
        """
        if self.config.projection_dim is None or features.shape[1] <= self.config.projection_dim:
            return features

        generator = torch.Generator().manual_seed(self.config.seed)
        projection = torch.randn(
            features.shape[1], self.config.projection_dim, generator=generator
        )
        projection = projection / projection.norm(dim=0, keepdim=True)
        return features @ projection

    @torch.no_grad()
    def select(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (N, D) patch embeddings.
        returns: (k, D) subset of `features` (original, unprojected values),
                 where k = ceil(N * sampling_ratio).
        """
        n = features.shape[0]
        k = max(1, int(n * self.config.sampling_ratio))

        proj = self._maybe_project(features)

        generator = torch.Generator().manual_seed(self.config.seed)
        start_idx = torch.randint(0, n, (1,), generator=generator).item()

        selected_indices = [start_idx]
        # min_dists[i] = distance from point i to nearest currently-selected point
        min_dists = torch.cdist(proj, proj[start_idx : start_idx + 1]).squeeze(1)

        for _ in range(1, k):
            next_idx = int(torch.argmax(min_dists).item())
            selected_indices.append(next_idx)
            new_dists = torch.cdist(proj, proj[next_idx : next_idx + 1]).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)

        selected = torch.tensor(selected_indices, dtype=torch.long)
        return features[selected]