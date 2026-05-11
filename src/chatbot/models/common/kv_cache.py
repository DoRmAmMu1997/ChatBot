"""KV-cache: stores the key/value tensors from past tokens during generation.

During training every token attends to every earlier token in parallel, so
there's no cache. During inference we generate one token at a time and we
DO NOT want to re-encode the whole prefix every step — the cache stores the
keys and values from previous steps and we append to it.

Two cache classes:

* :class:`KVCache` — standard per-layer (key, value) tensors. Shape
  ``[batch, kv_heads, seq, head_dim]``.
* :class:`MLAKVCache` — for Multi-head Latent Attention (Forge). Stores a
  *compressed* latent representation that is ~6–10x smaller than the
  equivalent standard KV cache. This is what makes Forge's 1M context
  practical in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class _LayerKV:
    key: torch.Tensor          # [batch, kv_heads, seq, head_dim]
    value: torch.Tensor        # [batch, kv_heads, seq, head_dim]


class KVCache:
    """Standard KV cache backed by a Python list of per-layer tensors."""

    def __init__(self, num_layers: int):
        self._layers: List[Optional[_LayerKV]] = [None] * num_layers

    def num_layers(self) -> int:
        return len(self._layers)

    def length(self, layer_idx: int = 0) -> int:
        """Number of cached tokens for the given layer (0 if empty)."""
        entry = self._layers[layer_idx]
        return 0 if entry is None else int(entry.key.shape[-2])

    def update(
        self,
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> _LayerKV:
        """Append new keys/values for this layer; return the full ``(K, V)``."""

        entry = self._layers[layer_idx]
        if entry is None:
            merged = _LayerKV(key=new_key, value=new_value)
        else:
            merged = _LayerKV(
                key=torch.cat([entry.key, new_key], dim=-2),
                value=torch.cat([entry.value, new_value], dim=-2),
            )
        self._layers[layer_idx] = merged
        return merged

    def truncate(self, max_length: int) -> None:
        """Crop the cache to the last ``max_length`` tokens (sliding window)."""

        for i, entry in enumerate(self._layers):
            if entry is None or entry.key.shape[-2] <= max_length:
                continue
            self._layers[i] = _LayerKV(
                key=entry.key[..., -max_length:, :],
                value=entry.value[..., -max_length:, :],
            )


@dataclass
class _LayerLatent:
    compressed_kv: torch.Tensor    # [batch, seq, kv_lora_rank]
    k_rope: torch.Tensor           # [batch, seq, qk_rope_head_dim]


class MLAKVCache:
    """Compressed KV cache for Multi-head Latent Attention.

    Instead of storing per-head keys/values (which scales with
    ``n_kv_heads * head_dim``), MLA stores a single low-rank latent
    plus a small RoPE-only component per token. The attention module
    decompresses on the fly during the score computation.
    """

    def __init__(self, num_layers: int):
        self._layers: List[Optional[_LayerLatent]] = [None] * num_layers

    def length(self, layer_idx: int = 0) -> int:
        entry = self._layers[layer_idx]
        return 0 if entry is None else int(entry.compressed_kv.shape[-2])

    def update(
        self,
        layer_idx: int,
        new_compressed_kv: torch.Tensor,
        new_k_rope: torch.Tensor,
    ) -> _LayerLatent:
        entry = self._layers[layer_idx]
        if entry is None:
            merged = _LayerLatent(compressed_kv=new_compressed_kv, k_rope=new_k_rope)
        else:
            merged = _LayerLatent(
                compressed_kv=torch.cat([entry.compressed_kv, new_compressed_kv], dim=-2),
                k_rope=torch.cat([entry.k_rope, new_k_rope], dim=-2),
            )
        self._layers[layer_idx] = merged
        return merged

    def truncate(self, max_length: int) -> None:
        for i, entry in enumerate(self._layers):
            if entry is None or entry.compressed_kv.shape[-2] <= max_length:
                continue
            self._layers[i] = _LayerLatent(
                compressed_kv=entry.compressed_kv[..., -max_length:, :],
                k_rope=entry.k_rope[..., -max_length:, :],
            )
