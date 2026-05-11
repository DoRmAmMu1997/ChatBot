"""SigLIP2-inspired Vision Transformer encoder.

The encoder is a stack of standard pre-norm Transformer blocks operating on
the patch tokens produced by ``PatchEmbed``. SigLIP differs from CLIP only
in training (sigmoid loss vs. softmax contrastive), so the encoder
architecture below works as a drop-in for either pre-trained backbone if
the user chooses to bootstrap from public weights.

We reuse the same ``RMSNorm`` and ``SwiGLU`` from the language stack and
implement a plain Multi-Head Attention specialized for image patches (no
RoPE here — images are 2-D and don't benefit from rotary positions; the
learnable 2-D position embeddings inside ``PatchEmbed`` are enough).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.ffn import SwiGLU
from ..common.normalization import RMSNorm
from .patch_embed import PatchEmbed


class _ViTAttention(nn.Module):
    """Plain multi-head attention used inside the ViT encoder."""

    def __init__(self, dim: int, num_heads: int, *, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # Vision encoders are NOT causal — every patch can attend to every other.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.o_proj(out)


class _ViTBlock(nn.Module):
    """One Transformer block of the vision encoder."""

    def __init__(self, dim: int, num_heads: int, ffn_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = _ViTAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder used by Aurora.

    Produces ``[B, num_patches, dim]`` patch features. The downstream
    :class:`MLPConnector` decides how many "soft image tokens" survive
    into the language model.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int = 3,
        dim: int = 1152,
        depth: int = 27,
        num_heads: int = 16,
        ffn_hidden: int | None = None,
    ):
        super().__init__()
        if ffn_hidden is None:
            ffn_hidden = dim * 4
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim)
        self.blocks = nn.ModuleList(
            [_ViTBlock(dim, num_heads, ffn_hidden) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
