"""RMSNorm — root-mean-square layer normalization.

Compared to the classic LayerNorm, RMSNorm:

  * drops the mean-subtraction (only normalizes by the *scale*),
  * drops the additive bias.

So instead of ``(x - mean) / std * gain + bias`` we just do
``x / rms(x) * gain``. This is cheaper and, in practice, works just as well
for very large language models (Llama 2/3, Mistral, DeepSeek, Gemma all use
RMSNorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Args:
        dim: hidden dimension to normalize across (the last axis of input).
        eps: small constant added to the variance for numerical stability;
            stops division-by-zero if the input has tiny magnitude.
    """

    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        # ``weight`` is sometimes called ``gain`` or ``scale``. It is the only
        # learnable parameter: per-feature multipliers that let the model
        # re-scale each dimension after normalization.
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We compute in float32 for stability even if the model is running in
        # bf16 — the squared mean of a bf16 tensor can underflow easily.
        original_dtype = x.dtype
        x_fp32 = x.float()

        # Root-mean-square along the feature dim. ``keepdim=True`` so the
        # divisor broadcasts cleanly against the original tensor.
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()

        # Multiply by the learned per-feature gain, then cast back so the
        # rest of the network keeps using the model's working dtype.
        return (x_fp32 * rms).to(original_dtype) * self.weight
