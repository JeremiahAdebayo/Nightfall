"""
Shared project configuration: the canonical MVTec category list and
checkpoint path conventions.

This module exists so that training, evaluation, calibration, and serving
can never silently drift apart on category names or checkpoint locations.
Previously these were duplicated (and diverged: `scripts/train.py` vs
`eval/train.py` vs a bare `from nightfall.config import ...`) across four modules.
"""

from __future__ import annotations

from pathlib import Path

ALL_MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


def checkpoint_path(output_dir: Path | str, category: str) -> Path:
    return Path(output_dir) / f"{category}_memory_bank.pt"
