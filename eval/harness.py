"""
Evaluation harness: runs a fitted PatchCore model against test data for
one or more categories, computes image AUROC, pixel AUROC, and PRO per
category, and assembles comparison tables against reference numbers
(v1's anomalib-based visual-defect-inspector results, and the original
PatchCore paper's published numbers).

Design note: reference numbers (v1, paper) are supplied by the caller as
plain dicts rather than hardcoded here, since they're external facts that
can change (e.g. if v1 gets re-benchmarked) and shouldn't be baked into
harness code. See scripts/ for where these are actually populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from core.patchcore import PatchCore
from eval.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro_score


@dataclass
class CategoryTestData:
    """Container for one category's test set."""

    category: str
    images: torch.Tensor  # (N, 3, H, W), preprocessed
    image_labels: np.ndarray  # (N,) 0=normal, 1=defective
    pixel_labels: np.ndarray  # (N, H, W) binary masks (0 for normal images)


@dataclass
class CategoryResult:
    category: str
    image_auroc: float
    pixel_auroc: float
    pro_score: float
    memory_bank_size: int
    num_test_images: int
    worst_scoring_indices: list = field(default_factory=list)


class EvalHarness:
    def __init__(self, model: PatchCore):
        self.model = model
        self.results: dict[str, CategoryResult] = {}

    def evaluate_category(
        self, test_data: CategoryTestData, num_failure_cases: int = 5
    ) -> CategoryResult:
        """
        Runs the model against one category's test set and computes all
        three metrics. Also records the indices of the worst-scoring
        defective images (lowest image_score among true positives, i.e.
        the defects the model was LEAST confident about) for the failure
        case analysis in the README.
        """
        scoring = self.model.predict(test_data.category, test_data.images)

        image_scores = scoring.image_score.detach().cpu().numpy()
        pixel_scores = scoring.pixel_map.detach().cpu().numpy()

        image_auroc = compute_image_auroc(image_scores, test_data.image_labels)
        pixel_auroc = compute_pixel_auroc(pixel_scores, test_data.pixel_labels)
        pro = compute_pro_score(pixel_scores, test_data.pixel_labels)

        # Failure cases: among truly defective images, which did the model
        # score lowest (i.e. was least confident were anomalous)? These are
        # the near-misses worth showing in the README failure analysis.
        defective_mask = test_data.image_labels == 1
        defective_indices = np.where(defective_mask)[0]
        defective_scores = image_scores[defective_mask]
        worst_order = np.argsort(defective_scores)  # ascending: worst first
        worst_indices = defective_indices[worst_order[:num_failure_cases]].tolist()

        result = CategoryResult(
            category=test_data.category,
            image_auroc=image_auroc,
            pixel_auroc=pixel_auroc,
            pro_score=pro,
            memory_bank_size=self.model.memory_bank_size(test_data.category),
            num_test_images=len(test_data.image_labels),
            worst_scoring_indices=worst_indices,
        )
        self.results[test_data.category] = result
        return result

    def aggregate(self) -> dict:
        """Mean metrics across all evaluated categories, unweighted."""
        if not self.results:
            return {}
        return {
            "image_auroc": float(np.mean([r.image_auroc for r in self.results.values()])),
            "pixel_auroc": float(np.mean([r.pixel_auroc for r in self.results.values()])),
            "pro_score": float(np.mean([r.pro_score for r in self.results.values()])),
            "num_categories": len(self.results),
        }

    def comparison_table(
        self,
        v1_numbers: dict[str, dict[str, float]] | None = None,
        paper_numbers: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """
        Builds a markdown table comparing Nightfall's per-category results
        against v1 (visual-defect-inspector, anomalib-based) and the
        original PatchCore paper, where available.

        v1_numbers / paper_numbers format:
            {"bottle": {"image_auroc": 1.0, "pixel_auroc": 0.978}, ...}
        Missing categories/metrics in either dict are rendered as "--".
        """
        v1_numbers = v1_numbers or {}
        paper_numbers = paper_numbers or {}

        header = (
            "| Category | Nightfall AUROC | Nightfall PRO | v1 AUROC | Paper AUROC |\n"
            "|---|---|---|---|---|\n"
        )
        rows = []
        for category, result in sorted(self.results.items()):
            v1_auroc = v1_numbers.get(category, {}).get("image_auroc")
            paper_auroc = paper_numbers.get(category, {}).get("image_auroc")
            rows.append(
                f"| {category} "
                f"| {result.image_auroc:.3f} "
                f"| {result.pro_score:.3f} "
                f"| {f'{v1_auroc:.3f}' if v1_auroc is not None else '--'} "
                f"| {f'{paper_auroc:.3f}' if paper_auroc is not None else '--'} |"
            )

        agg = self.aggregate()
        if agg:
            rows.append(
                f"| **mean ({agg['num_categories']} categories)** "
                f"| **{agg['image_auroc']:.3f}** "
                f"| **{agg['pro_score']:.3f}** | -- | -- |"
            )

        return header + "\n".join(rows)