"""
Image preprocessing for PatchCore.

Why this lives in its own module rather than inside PatchCore or the
feature extractor (see DECISIONS.md):

PatchFeatureExtractor assumes its input is already a normalized tensor of
the correct size -- it has no opinion on how you got there, which keeps it
testable with synthetic tensors and reusable in contexts that don't start
from image files (e.g. tests, serving code that already has decoded
tensors). Preprocessing is a separate concern with its own failure modes
(wrong normalization stats, wrong resize interpolation) that deserve their
own module and their own tests, rather than being buried as a side effect
of model construction.

Normalization: WideResNet50's ImageNet-pretrained weights were trained on
images normalized with ImageNet's channel-wise mean/std. Using different
(or no) normalization doesn't raise an error -- the model still runs and
produces *a* feature map -- but the features are systematically shifted
from the distribution the backbone was trained on, which silently degrades
feature quality and, downstream, AUROC. This is exactly the kind of bug
that doesn't announce itself; it just looks like "the model isn't very
good," so it's worth a dedicated regression test (see tests/).

Resize: MVTec images are typically much larger than the 224x224-ish
resolution ImageNet backbones expect efficient behavior at. We resize to
a configurable target size (256 default, matching common PatchCore repro
settings) via bilinear interpolation, then center-crop to the exact target
if the aspect ratio isn't square -- avoiding distortion from a naive
resize-to-square that would stretch defects out of their true proportions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


@dataclass
class PreprocessConfig:
    resize_size: int = 256
    crop_size: int = 224
    # ImageNet channel-wise normalization stats -- required because the
    # WideResNet50 backbone is ImageNet-pretrained (see module docstring).
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImagePreprocessor:
    """
    Converts raw images (file paths, PIL Images, or already-loaded arrays)
    into normalized tensors ready for PatchFeatureExtractor.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._transform = transforms.Compose(
            [
                transforms.Resize(self.config.resize_size),
                transforms.CenterCrop(self.config.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std),
            ]
        )

    def __call__(self, image: Image.Image | str | Path) -> torch.Tensor:
        """
        Returns a single (3, crop_size, crop_size) normalized tensor.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")

        return self._transform(image)

    def batch(self, images: list[Image.Image | str | Path]) -> torch.Tensor:
        """
        Returns a (N, 3, crop_size, crop_size) batch tensor from a list of
        images. All images are processed independently then stacked --
        this assumes a uniform crop_size, which is enforced by construction
        since every image goes through the same CenterCrop.
        """
        return torch.stack([self(img) for img in images], dim=0)

    def inverse_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Undoes the ImageNet normalization for visualization purposes
        (e.g. overlaying a pixel-level anomaly heatmap on the original
        image). Does NOT undo the resize/crop -- callers should keep
        track of the original image separately if they need it.
        """
        mean = torch.tensor(self.config.mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(self.config.std, device=tensor.device).view(-1, 1, 1)
        return (tensor * std + mean).clamp(0, 1)