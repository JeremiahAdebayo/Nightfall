"""
Public core API.

Attributes are resolved lazily (PEP 562): importing `nightfall.core` should
not force `torchvision`/`onnxruntime` to be importable just to reach, say,
`MemoryBank` or `GreedyCoresetSampler`. Each submodule still imports its own
heavy dependencies normally when actually used.
"""

from __future__ import annotations

__all__ = [
    "PatchFeatureExtractor",
    "GreedyCoresetSampler",
    "MemoryBank",
    "MemoryBankConfig",
    "PatchCore",
    "ImagePreprocessor",
    "PreprocessConfig",
    "OnnxFeatureExtractor",
    "ExtractorConfig",
    "CoresetConfig",
]

_EXPORTS = {
    "PatchFeatureExtractor": "feature_extractor",
    "ExtractorConfig": "feature_extractor",
    "GreedyCoresetSampler": "coreset",
    "CoresetConfig": "coreset",
    "MemoryBank": "memory_bank",
    "MemoryBankConfig": "memory_bank",
    "PatchCore": "patchcore",
    "ImagePreprocessor": "preprocessing",
    "PreprocessConfig": "preprocessing",
    "OnnxFeatureExtractor": "onnx_feature_extractor",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module_name}"), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
