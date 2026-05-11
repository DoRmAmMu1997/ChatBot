"""Connect the vision tower's output to the language model's embedding space.

Pattern follows LLaVA-NeXT / Cambrian: a small MLP projects per-patch vision
features into the same dimension as the LLM's token embeddings, so the
patches can be treated as ordinary tokens by the language stack.

We also support optional **pooling** — reducing the number of visual tokens
before they enter the LLM, e.g. to fit more images in context. Default is
no pooling.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPConnector(nn.Module):
    """Project per-patch vision features into the LLM's embedding space.

    Args:
        vision_dim: dim of the vision encoder's output.
        hidden_dim: intermediate dim of the projector (matches the LLM's
            ``d_model`` usually).
        llm_dim: dim of the language model's token embeddings.
        pool: if set, reduce visual tokens by this factor along each grid axis.
    """

    def __init__(
        self,
        vision_dim: int,
        hidden_dim: int,
        llm_dim: int,
        *,
        pool: Optional[int] = None,
    ):
        super().__init__()
        self.pool = pool
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """``[B, N, vision_dim] → [B, N', llm_dim]`` (N' depends on pooling)."""

        if self.pool and self.pool > 1:
            vision_tokens = self._average_pool(vision_tokens, self.pool)
        return self.fc2(self.act(self.fc1(vision_tokens)))

    @staticmethod
    def _average_pool(x: torch.Tensor, pool: int) -> torch.Tensor:
        # Assume the patch tokens form a square grid. Reshape to grid, pool,
        # then flatten back.
        batch, n, dim = x.shape
        side = int(n ** 0.5)
        if side * side != n:
            return x  # not square; skip pooling rather than crash
        grid = x.transpose(1, 2).reshape(batch, dim, side, side)
        grid = F.avg_pool2d(grid, kernel_size=pool, stride=pool)
        return grid.flatten(2).transpose(1, 2)
