"""RoPE (Rotary Position Embedding) plus YaRN long-context scaling.

RoPE encodes "where" a token sits in the sequence by rotating the query and
key vectors by an angle proportional to the token's position. The rotation
preserves dot-product geometry while giving the attention mechanism a
position-aware signal.

For a head dim of size ``D`` (must be even), we split each vector into pairs
``(x_{2i}, x_{2i+1})`` and apply a 2D rotation by angle
``theta_i(pos) = pos * base^{-2i/D}``.

YaRN (Yet Another RoPE eNcoding) extends RoPE to longer contexts than the
model was trained on by re-scaling the angles in a frequency-aware way. We
use a simplified two-piece scheme: high-frequency dims keep their old
angles, low-frequency dims get stretched by the requested factor.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _build_inv_freq(head_dim: int, base: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Frequencies used by RoPE: theta_i = base^{-2i / head_dim}."""

    half = head_dim // 2
    idx = torch.arange(0, half, device=device, dtype=dtype)
    return 1.0 / (base ** (2 * idx / head_dim))


class RotaryEmbedding(nn.Module):
    """Standard RoPE, no scaling.

    Pre-computes a cos/sin table up to ``max_position_embeddings``. Tables
    grow lazily if a longer context shows up at inference time.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head dim must be even"
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Buffers (not parameters) — they aren't trained, just precomputed.
        self.register_buffer("inv_freq", _build_inv_freq(head_dim, base), persistent=False)
        cos, sin = self._compute_cos_sin(max_position_embeddings, self.inv_freq.device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _compute_cos_sin(self, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # Outer product positions × frequencies → [seq, half] angles.
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        # Duplicate so the final dim matches ``head_dim`` after the rotation.
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def _ensure_capacity(self, seq_len: int, device) -> None:
        if seq_len <= self.cos_cached.shape[0] and self.cos_cached.device == device:
            return
        cos, sin = self._compute_cos_sin(seq_len, device)
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(self, seq_len: int, device, dtype=torch.float32, offset: int = 0):
        """Return ``(cos, sin)`` slices of shape ``[seq, head_dim]``."""
        self._ensure_capacity(offset + seq_len, device)
        cos = self.cos_cached[offset : offset + seq_len].to(dtype)
        sin = self.sin_cached[offset : offset + seq_len].to(dtype)
        return cos, sin


class YarnRotaryEmbedding(RotaryEmbedding):
    """RoPE with YaRN extrapolation for long contexts.

    The trick: keep the high-frequency components (which carry local
    position) unchanged, and stretch only the low-frequency components
    (which carry long-range position) by ``factor``. A linear ramp blends
    them at the boundary so there's no jump in attention.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        *,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        # We override inv_freq before super().__init__ pre-computes the cache.
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.max_position_embeddings = max_position_embeddings
        nn.Module.__init__(self)
        inv_freq = self._build_yarn_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._compute_cos_sin(max_position_embeddings, inv_freq.device)
        # Cosine/sine are extra-scaled by a small extension factor — this is
        # the "ATTENTION_SCALE" trick from the YaRN paper that compensates
        # for the inflated softmax temperature at extreme positions.
        attention_scale = 0.1 * math.log(scaling_factor) + 1.0 if scaling_factor > 1 else 1.0
        self.register_buffer("cos_cached", cos * attention_scale, persistent=False)
        self.register_buffer("sin_cached", sin * attention_scale, persistent=False)

    def _build_yarn_inv_freq(self) -> torch.Tensor:
        head_dim = self.head_dim
        base = self.base
        factor = self.scaling_factor
        original_max = self.original_max
        beta_fast, beta_slow = self.beta_fast, self.beta_slow

        # Standard inverse frequencies (the un-scaled baseline).
        base_inv = _build_inv_freq(head_dim, base)

        # YaRN works in wavelengths. Convert frequency to wavelength.
        wavelengths = 2 * math.pi / base_inv

        # Compute the "ramp" — for each frequency, how much of the
        # extrapolated (stretched) variant to mix in vs. the original.
        def correction(num_rotations: float) -> float:
            return (
                head_dim
                * math.log(original_max / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction(beta_slow)), 0)
        high = min(math.ceil(correction(beta_fast)), head_dim // 2 - 1)

        if low == high:
            high += 0.001  # avoid div-by-zero in the linear ramp

        ramp_indices = torch.arange(head_dim // 2, dtype=torch.float32)
        ramp = (ramp_indices - low) / (high - low)
        ramp = torch.clamp(ramp, 0.0, 1.0)

        # 1 - ramp = "keep original"; ramp = "use stretched". So:
        #   inv_freq_yarn = (1 - ramp) * original  +  ramp * (original / factor)
        # For longer contexts factor > 1, so we lengthen the wavelengths
        # of the low-frequency components, letting them encode positions
        # far beyond the original training length.
        inv_freq_extrapolated = base_inv  # high-freq dims, unchanged
        inv_freq_interpolated = base_inv / factor  # low-freq dims, stretched
        inv_freq = inv_freq_extrapolated * (1 - ramp) + inv_freq_interpolated * ramp
        return inv_freq


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the feature dim by 90 degrees.

    Splitting the last dim into two halves and computing
    ``[-x2, x1]`` is mathematically equivalent to applying a 2D rotation
    matrix to interleaved pairs. The split-half form is friendlier to
    vectorized ops and matches the formula in the Llama / DeepSeek codebases.
    """

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to query and key tensors.

    Args:
        q, k: shape ``[batch, n_heads, seq, head_dim]``.
        cos, sin: shape ``[seq, head_dim]`` (or ``[batch, seq, head_dim]``).
        position_ids: optional ``[batch, seq]`` so prefixes can use offsets.

    Returns: rotated ``(q, k)`` of the same shapes.
    """

    if position_ids is not None:
        # Gather cos/sin per-position-id (used for packed sequences).
        cos = cos[position_ids]
        sin = sin[position_ids]
    # Broadcast over heads: [seq, head_dim] → [1, 1, seq, head_dim]
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def _scaling_attr(scaling_cfg, key, default):
    """Read a field from either a dict-like config or a dataclass instance.

    Configs reach us in two shapes: an OmegaConf ``DictConfig`` (uses
    ``.get``) and a strongly-typed dataclass (uses attribute access). One
    tiny helper hides the difference so the caller doesn't care.
    """

    if scaling_cfg is None:
        return default
    if hasattr(scaling_cfg, "get") and callable(scaling_cfg.get):
        return scaling_cfg.get(key, default)
    return getattr(scaling_cfg, key, default)


def build_rotary_embedding(config) -> RotaryEmbedding:
    """Construct the right RoPE variant from a model config block."""

    head_dim = int(config.head_dim)
    max_pos = int(config.max_position_embeddings)
    base = float(getattr(config, "rope_base", 10000.0))
    scaling_cfg = getattr(config, "rope_scaling", None)

    scaling_type = str(_scaling_attr(scaling_cfg, "type", "none")).lower()
    if scaling_cfg is None or scaling_type == "none":
        return RotaryEmbedding(head_dim=head_dim, max_position_embeddings=max_pos, base=base)
    factor = float(_scaling_attr(scaling_cfg, "factor", 1.0))
    if scaling_type == "linear":
        # Linear scaling is RoPE with positions divided by ``factor``.
        return RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            base=base * factor,
        )
    if scaling_type == "yarn":
        original_max = int(_scaling_attr(
            scaling_cfg,
            "original_max_position_embeddings",
            max_pos // int(max(factor, 1)),
        ))
        return YarnRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            base=base,
            scaling_factor=factor,
            original_max_position_embeddings=original_max,
        )
    raise ValueError(f"Unknown rope_scaling.type: {scaling_type!r}")
