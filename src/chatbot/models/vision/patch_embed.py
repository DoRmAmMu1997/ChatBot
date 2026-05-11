"""Cut an image into patches and embed each one as a vector.

A Vision Transformer treats an image like a sequence of "visual tokens":
each token is one fixed-size patch (e.g. 14×14 pixels). This module:

  1. Slices an ``[B, 3, H, W]`` image into non-overlapping patches.
  2. Projects each patch (a 14*14*3 = 588-D vector) through a learnable
     linear layer to the ViT's hidden dimension.
  3. Adds learnable 2-D position embeddings so the encoder knows where each
     patch was on the grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Cut H×W image into ``(H/p)x(W/p)`` patches and embed them."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        # A strided conv with kernel = stride = patch_size is identical to
        # "linear-embed every patch independently". This is the standard ViT
        # patch-embed trick — it's a Conv2d only by convenience.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable position embeddings, one per patch.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Args: ``[B, 3, H, W]`` images. Returns ``[B, num_patches, embed_dim]``."""

        # Convolve to ``[B, embed_dim, grid, grid]``, then flatten the grid
        # into a sequence of patch tokens.
        x = self.proj(images)
        batch, embed_dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        # If the input was a different resolution than ``image_size``, the
        # patch count may differ — we interpolate the position embeddings
        # rather than crash. Production multimodal models do this so users
        # can pass non-default-size images without retraining.
        if x.shape[1] != self.pos_embed.shape[1]:
            x = x + self._resample_pos_embed(self.pos_embed, h, w)
        else:
            x = x + self.pos_embed
        return x

    def _resample_pos_embed(self, pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # ``pos_embed`` is shape [1, N, D]. Re-arrange to a 2-D grid, resize
        # with bilinear interpolation, then flatten again.
        n, d = pos_embed.shape[1], pos_embed.shape[2]
        side = int(math.sqrt(n))
        grid = pos_embed.reshape(1, side, side, d).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(1, h * w, d)
