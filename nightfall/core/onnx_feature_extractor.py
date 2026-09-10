"""
ONNX Runtime-backed drop-in replacement for PatchFeatureExtractor.

Extracted from Nightfall.ipynb (Phase 3 quantization experiments), where
it was previously defined inline per-cell and duplicated across cells.

This class implements the same `extract_patch_vectors(x)` signature the
PyTorch extractor exposes, so PatchCore, MemoryBank, and EvalHarness are
all unmodified when it's swapped in: they just call
extractor.extract_patch_vectors().

Inference runs in batches to avoid holding the whole set's activations in
memory at once. The last batch may be smaller than batch_size -- Python
slicing handles this safely (yields whatever remains).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class OnnxFeatureExtractor:
    def __init__(
        self,
        onnx_path: Path | str,
        batch_size: int = 16,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self.batch_size = batch_size

    def extract_patch_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) normalized image batch.
        returns: (vectors, (B, H', W')) where vectors is (B * H' * W', C_total)
            flattened patch embeddings -- matching the PyTorch extractor's
            extract_patch_vectors() return signature.
        """
        vectors_list = []
        spatial_shape = None
        for i in range(0, x.shape[0], self.batch_size):
            batch = x[i : i + self.batch_size].detach().cpu().numpy().astype(np.float32)
            fmap = self.session.run([self._output_name], {self._input_name: batch})[0]
            fmap = torch.from_numpy(fmap)
            b, c, h, w = fmap.shape
            if spatial_shape is None:
                spatial_shape = (b, h, w)
            vectors_list.append(fmap.permute(0, 2, 3, 1).reshape(b * h * w, c))
        return torch.cat(vectors_list, dim=0), spatial_shape
