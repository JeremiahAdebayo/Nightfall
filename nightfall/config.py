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


def int8_checkpoint_path(output_dir: Path | str, category: str) -> Path:
    """
    Checkpoint for a memory bank refit with the INT8 ONNX extractor
    (scripts/run_eval_int8.py --refit-bank). Deliberately a separate file
    from the fp32 bank so the mismatched-bank baseline's checkpoints and
    the consistent-pipeline refit checkpoints can never overwrite each
    other.
    """
    return Path(output_dir) / f"{category}_memory_bank_int8.pt"
