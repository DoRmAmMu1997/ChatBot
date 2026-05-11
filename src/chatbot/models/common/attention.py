"""Attention modules: GQA (used by Aurora) and MLA (used by Forge).

Both modules expose the same call signature so the surrounding decoder
block doesn't care which variant is plugged in:

    out = attn(x, cos, sin, attention_mask=..., kv_cache=..., layer_idx=...)

We delegate the actual softmax(qkᵀ/√d)·V math to
``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which on
CUDA dispatches to flash-attention 2/3 when available, falling back to
a memory-efficient PyTorch kernel otherwise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache, MLAKVCache
from .rope import apply_rotary_emb


def _repeat_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    """Expand a KV tensor so it has the same head count as Q.

    In GQA the number of KV heads is smaller than the number of Q heads.
    Each KV head is shared by ``repeat`` query heads. We just repeat the
    tensor along the head dim; PyTorch's SDPA expects matching head counts.
    """

    if repeat == 1:
        return x
    batch, kv_heads, seq, head_dim = x.shape
    return (
        x.unsqueeze(2)
        .expand(batch, kv_heads, repeat, seq, head_dim)
        .reshape(batch, kv_heads * repeat, seq, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """GQA — query heads outnumber key/value heads.

    Why GQA: full multi-head attention is parameter-symmetric in Q, K, V.
    GQA keeps the same number of Q heads (so capacity per token is the
    same) but shrinks K and V to far fewer heads. This saves a huge amount
    of KV-cache memory at long context with negligible quality loss.

    Args:
        d_model: residual dim.
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads (must divide n_heads).
        head_dim: per-head dimension.
        attn_dropout: dropout on attention weights (off by default at scale).
        bias: include bias on the projections (almost always False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout

        # Q is the full-width projection; K/V are narrower (grouped).
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin, position_ids=position_ids)

        # Cache management — only used during incremental generation.
        if kv_cache is not None:
            merged = kv_cache.update(layer_idx, k, v)
            k, v = merged.key, merged.value

        # Bring KV head count up to Q head count for SDPA.
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        # If we have an explicit mask, use it; otherwise rely on SDPA's
        # built-in ``is_causal`` flag (cheaper, no allocation).
        if attention_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=kv_cache is None,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V3 style).

    Idea: factorize Q and KV through low-rank latent spaces so the KV cache
    only stores the latent, not the full per-head projections.

    For each token we keep:

      * ``compressed_kv``: latent of dim ``kv_lora_rank`` (≈ 512 in Forge).
      * ``k_rope``: a small RoPE-only component of dim ``qk_rope_head_dim``
        (≈ 64) so positional info isn't lost.

    Total cached state per token = ``kv_lora_rank + qk_rope_head_dim``
    ≈ 576 floats vs. a plain MHA cache of
    ``2 * n_kv_heads * head_dim`` ≈ 2048 floats. At 1M context that
    difference is hundreds of GB.

    The Q side is also low-rank for a smaller training cost, though Q has
    no cache so it doesn't affect inference memory.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        attn_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attn_dropout = attn_dropout

        # Q: low-rank factorization through a latent.
        self.q_a_proj = nn.Linear(d_model, q_lora_rank, bias=bias)
        self.q_a_norm = nn.LayerNorm(q_lora_rank)
        # The "B" projection produces per-head Q including BOTH the nope
        # (un-rotated) and the rope (rotated) parts in one tensor.
        self.q_b_proj = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=bias)

        # KV: the entire cache content goes through here. We split into a
        # latent (cached) and a small key-rope (also cached). Values and the
        # nope part of keys are re-expanded from the latent on demand.
        self.kv_a_proj_with_mqa = nn.Linear(
            d_model, kv_lora_rank + qk_rope_head_dim, bias=bias
        )
        self.kv_a_norm = nn.LayerNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=bias
        )

        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=bias)

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        # MLA uses a custom softmax scale: 1 / sqrt(qk_head_dim) by default.
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[MLAKVCache] = None,
        layer_idx: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape

        # ---- Q side ----
        # Q passes through a low-rank bottleneck: ``x → q_a_proj``
        # (d_model → q_lora_rank) → LayerNorm → ``q_b_proj`` (q_lora_rank
        # → n_heads × qk_head_dim). The bottleneck makes Q cheaper to
        # train without affecting inference (Q isn't cached).
        q_latent = self.q_a_norm(self.q_a_proj(x))
        q = self.q_b_proj(q_latent).view(batch, seq, self.n_heads, self.qk_head_dim).transpose(1, 2)
        # Each head's Q is split into two pieces:
        #   * ``q_nope`` — the "no positional encoding" half. Used as-is.
        #   * ``q_rope`` — the half that *will* receive RoPE rotation.
        # Splitting like this means we only rotate a small slice of Q (and
        # the matching slice of K), which saves compute at long contexts.
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # ---- KV side: compute the *new* compressed latent + k-rope ----
        # One Linear projects x to (latent || rope-key). The latent is the
        # whole "value" of the KV cache for this token — we'll re-expand
        # it on demand into per-head keys (nope-part) and values.
        kv_a = self.kv_a_proj_with_mqa(x)
        compressed_kv_new, k_rope_new = torch.split(
            kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv_new = self.kv_a_norm(compressed_kv_new)
        # k_rope is shared across heads (no head dim in the cache); add a
        # broadcast dim for the attention computation later. This is the
        # second cached tensor (the first is ``compressed_kv``).
        k_rope_new = k_rope_new.unsqueeze(2)  # [batch, seq, 1, qk_rope_head_dim]

        # ---- Append to the cache (or build it fresh) ----
        if kv_cache is not None:
            merged = kv_cache.update(layer_idx, compressed_kv_new, k_rope_new.squeeze(2))
            compressed_kv = merged.compressed_kv         # [batch, total_seq, kv_lora_rank]
            k_rope = merged.k_rope.unsqueeze(2)          # [batch, total_seq, 1, qk_rope_head_dim]
        else:
            compressed_kv = compressed_kv_new
            k_rope = k_rope_new

        # ---- Re-expand the latent to per-head K (nope) and V ----
        # One Linear maps the cached latent back up to the full per-head
        # K and V dimensions. This is the "B" half of the LoRA-style
        # factorization, mirroring Q. Crucially the *cache* stores only
        # the small ``compressed_kv``; the wide re-expansion is recomputed
        # at every step from the cache.
        kv_b = self.kv_b_proj(compressed_kv)
        total_seq = compressed_kv.shape[1]
        kv_b = kv_b.view(batch, total_seq, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(
            kv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # Transpose to standard ``[batch, heads, seq, dim]`` layout that
        # ``scaled_dot_product_attention`` expects.
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)
        # k_rope was stored shared-across-heads to save cache; here we
        # broadcast it to every head so the dimensions line up with k_nope.
        k_rope = k_rope.expand(batch, total_seq, self.n_heads, self.qk_rope_head_dim).transpose(1, 2)

        # ---- Apply RoPE to the small rope-only pieces of Q and K ----
        # Only the rope-halves get rotated. The nope-halves carry content
        # information that should be position-agnostic; mixing RoPE in
        # there would entangle "what" with "where".
        q_rope, k_rope_rotated = apply_rotary_emb(q_rope, k_rope, cos, sin, position_ids=position_ids)

        # ---- Re-assemble Q and K by concatenating the two halves ----
        # Final per-head Q and K each have width ``qk_head_dim`` =
        # nope_dim + rope_dim. SDPA from here is identical to regular
        # multi-head attention.
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope_rotated], dim=-1)

        if attention_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=kv_cache is None,
                scale=self.softmax_scale,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.softmax_scale,
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.v_head_dim)
        return self.o_proj(out)


def _attn_attr(attn_cfg, key, default=None):
    """Read a field from either a ``DictConfig``/dict or a dataclass instance."""

    if attn_cfg is None:
        return default
    if hasattr(attn_cfg, "get") and callable(attn_cfg.get):
        return attn_cfg.get(key, default)
    return getattr(attn_cfg, key, default)


def build_attention(cfg) -> nn.Module:
    """Pick the right attention module from a model config.

    Accepts either an OmegaConf ``DictConfig`` (dict-shaped) or a strongly
    typed dataclass (`ForgeConfig`, `AuroraConfig`). The helper above
    normalises field access so we don't need a branch here.
    """

    attn_cfg = getattr(cfg, "attention", None)
    variant = str(_attn_attr(attn_cfg, "variant", "gqa")).lower()

    if variant == "gqa":
        return GroupedQueryAttention(
            d_model=int(cfg.d_model),
            n_heads=int(cfg.n_heads),
            n_kv_heads=int(cfg.n_kv_heads),
            head_dim=int(cfg.head_dim),
        )
    if variant == "mla":
        return MultiHeadLatentAttention(
            d_model=int(cfg.d_model),
            n_heads=int(cfg.n_heads),
            q_lora_rank=int(_attn_attr(attn_cfg, "q_lora_rank", 0)),
            kv_lora_rank=int(_attn_attr(attn_cfg, "kv_lora_rank", 0)),
            qk_nope_head_dim=int(_attn_attr(attn_cfg, "qk_nope_head_dim", 0)),
            qk_rope_head_dim=int(_attn_attr(attn_cfg, "qk_rope_head_dim", 0)),
            v_head_dim=int(_attn_attr(attn_cfg, "v_head_dim", 0)),
        )
    raise ValueError(f"Unknown attention variant {variant!r}")
