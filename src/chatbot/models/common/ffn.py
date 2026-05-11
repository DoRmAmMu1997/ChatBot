"""Feed-forward networks (FFN) for our Transformer blocks.

We only ship one variant: SwiGLU, the gated SiLU activation used by Llama
and most modern open LLMs. The math is:

    FFN(x) = ( SiLU(x W_gate) * (x W_up) ) W_down

The gate × up product is what makes this "gated": one branch decides how
much of the other branch to pass through. Empirically beats plain ReLU /
GELU FFNs at scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Args:
        dim: hidden dimension of the residual stream (``d_model``).
        hidden: inner FFN dimension (typically ~3x ``d_model``).
        bias: whether to include bias on each linear (LLMs usually don't).
    """

    def __init__(self, dim: int, hidden: int, *, bias: bool = False):
        super().__init__()
        # Two parallel projections feeding into a gate. Sometimes you'll see
        # them fused into a single ``Linear(dim, 2 * hidden)`` for speed,
        # but two separate Linears keep the param-naming obvious for LoRA
        # target_modules selection.
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        # Projection back to the residual stream dimension.
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU = x * sigmoid(x). It's smooth and unbounded above, which
        # helps avoid the "dying ReLU" problem at scale.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
