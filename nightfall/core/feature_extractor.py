"""
Patch-level feature extraction for PatchCore.

Design notes (see DECISIONS.md for full rationale):

- We hook mid-level layers (layer2, layer3 of WideResNet50) rather than the
  final pooled embedding. Early layers are too low-level (edges/textures,
  not defect-discriminative); the final layer is too semantically coarse
  and loses the spatial resolution we need for pixel-level anomaly maps.
  This mirrors the original PatchCore paper's choice.

- "Locally aware" patch features: each spatial location's feature vector
  is replaced by an aggregate (mean) over its local neighborhood via
  average pooling. This gives each patch feature some receptive-field
  context beyond a single pixel of the feature map, which measurably
  improves robustness to small localization noise -- without this step,
  coreset selection tends to pick redundant near-duplicate patches from
  noisy single-pixel activations.

- Multi-scale fusion: layer2 and layer3 feature maps are at different
  spatial resolutions. We resize layer3 up to layer2's resolution
  (bilinear interpolation) and concatenate channel-wise, rather than
  picking a single layer. This trades some memory for meaningfully
  better localization of small defects (layer2 alone under-detects
  small anomalies; layer3 alone over-smooths their boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


@dataclass
class ExtractorConfig:
    layers: Sequence[str] = ("layer2", "layer3")
    local_pool_kernel: int = 3
    local_pool_stride: int = 1
    device: str = "cpu"


class PatchFeatureExtractor(nn.Module):
    """
    Wraps a frozen WideResNet50 backbone and exposes patch-level,
    locally-aware, multi-scale feature maps.

    Output of `forward`: tensor of shape (B, C_total, H, W) where H, W
    match the coarsest of the selected layers (layer2's resolution when
    using layer2+layer3), and C_total is the channel-concatenation of
    all selected layers.
    """

    def __init__(self, config: ExtractorConfig | None = None):
        super().__init__()
        self.config = config or ExtractorConfig()

        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        self._features: dict[str, torch.Tensor] = {}
        self._hooks = []
        for layer_name in self.config.layers:
            layer = getattr(backbone, layer_name)
            handle = layer.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(handle)

        self.backbone = backbone
        self.local_pool = nn.AvgPool2d(
            kernel_size=self.config.local_pool_kernel,
            stride=self.config.local_pool_stride,
            padding=self.config.local_pool_kernel // 2,
        )
        self.to(self.config.device)

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self._features[name] = output

        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) normalized image batch.
        returns: (B, C_total, H', W') patch feature map.
        """
        self._features.clear()
        _ = self.backbone(x)

        layer_maps = [self._features[name] for name in self.config.layers]

        # Locally-aware pooling per layer, before resizing/fusion.
        layer_maps = [self.local_pool(fm) for fm in layer_maps]

        # Resize all maps to the resolution of the first (coarsest-stride,
        # highest-resolution) selected layer.
        target_hw = layer_maps[0].shape[-2:]
        resized = [layer_maps[0]]
        for fm in layer_maps[1:]:
            resized.append(
                F.interpolate(fm, size=target_hw, mode="bilinear", align_corners=False)
            )

        fused = torch.cat(resized, dim=1)  # channel-wise concat
        return fused

    def extract_patch_vectors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper: returns patch features flattened to
        (B * H' * W', C_total) -- i.e. one row per spatial patch,
        ready for memory bank insertion or scoring.
        """
        fmap = self.forward(x)
        b, c, h, w = fmap.shape
        return fmap.permute(0, 2, 3, 1).reshape(b * h * w, c), (b, h, w)

    def __del__(self):
        for handle in getattr(self, "_hooks", []):
            handle.remove()