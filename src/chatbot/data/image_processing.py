"""Image preprocessing for Aurora's vision tower.

A vision encoder expects images to be uniformly sized and normalized. This
module ships a small, dependency-light preprocessor:

  * resize the shorter side to ``image_size`` while preserving aspect ratio,
  * center-crop to square,
  * convert to a torch tensor in ``[0, 1]``,
  * apply standard ImageNet-style mean/std normalization.

PIL is the only outside dependency. We keep things vanilla so the user can
swap in their own pipeline (e.g. SigLIP2's exact preprocessing) easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from PIL import Image


# Same normalization constants as CLIP / SigLIP.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


@dataclass
class ImagePreprocessor:
    image_size: int = 384
    mean: Sequence[float] = DEFAULT_MEAN
    std: Sequence[float] = DEFAULT_STD

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        # Resize so the shorter side equals ``image_size``.
        w, h = image.size
        if w < h:
            new_w = self.image_size
            new_h = int(round(h * self.image_size / w))
        else:
            new_h = self.image_size
            new_w = int(round(w * self.image_size / h))
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # Center crop to a perfect square.
        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        image = image.crop((left, top, left + self.image_size, top + self.image_size))

        # PIL → tensor in [0, 1].
        tensor = torch.tensor(list(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        # Per-channel normalization.
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor


def batch_images(images: List[Image.Image], preprocessor: ImagePreprocessor) -> torch.Tensor:
    """Preprocess a list of PIL images into a single ``[N, 3, H, W]`` tensor."""

    return torch.stack([preprocessor(img) for img in images], dim=0)
