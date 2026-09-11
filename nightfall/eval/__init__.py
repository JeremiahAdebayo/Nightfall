"""
Public evaluation API, resolved lazily (PEP 562).

`nightfall.eval.metrics` is pure numpy/scipy/sklearn, while the harness and
dataloader pull in the model stack (and therefore torchvision). Eager
re-exports here meant you could not compute a metric -- or run its tests --
without the heavy training dependencies being installed.
"""

from __future__ import annotations

__all__ = [
    "compute_image_auroc",
    "compute_pixel_auroc",
    "compute_pro_score",
    "EvalHarness",
    "CategoryResult",
    "CategoryTestData",
    "load_category_test_data",
    "geometry_metrics",
]

_EXPORTS = {
    "compute_image_auroc": "metrics",
    "compute_pixel_auroc": "metrics",
    "compute_pro_score": "metrics",
    "EvalHarness": "harness",
    "CategoryResult": "harness",
    "CategoryTestData": "harness",
    "load_category_test_data": "dataloader",
    "geometry_metrics": "quantization_geometry",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module_name}"), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
