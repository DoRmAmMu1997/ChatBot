"""A single decoder block: ``RMSNorm → Attention → residual → RMSNorm → FFN/MoE → residual``.

Pre-norm is used everywhere because it trains more stably than post-norm at
scale: every residual path is the identity in the limit of small weight
norms, which keeps gradients well-behaved through dozens of layers.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention, MultiHeadLatentAttention
from .ffn import SwiGLU
from .kv_cache import KVCache, MLAKVCache
from .moe import MixtureOfExperts, MoEOutput
from .normalization import RMSNorm


class DecoderBlock(nn.Module):
    """One Transformer decoder layer.

    The block is parametric in:
      * which attention module to use (GQA or MLA), and
      * which feed-forward path to use (dense SwiGLU or MoE).

    The Aurora and Forge model classes assemble lists of these blocks with
    different configurations per layer (e.g. Forge has dense FFNs in its
    first three layers and MoE after that).
    """

    def __init__(
        self,
        *,
        d_model: int,
        attn: Union[GroupedQueryAttention, MultiHeadLatentAttention],
        ffn: Union[SwiGLU, MixtureOfExperts],
        rms_eps: float = 1.0e-6,
    ):
        super().__init__()
        self.input_norm = RMSNorm(d_model, eps=rms_eps)
        self.attn = attn
        self.post_attn_norm = RMSNorm(d_model, eps=rms_eps)
        self.ffn = ffn

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Union[KVCache, MLAKVCache]] = None,
        layer_idx: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[MoEOutput]]:
        # ---- Attention sub-layer ----
        # 1) Stash the input as the "residual" so we can add it back later.
        #    The residual connection (`x = residual + ...`) is what keeps the
        #    gradient signal alive through a 64-layer-deep network — without
        #    it, training a deep Transformer is wildly unstable.
        # 2) Normalize before attention (pre-norm). Pre-norm trains far more
        #    stably than post-norm at scale, which is why every modern LLM
        #    uses it.
        # 3) Run attention. The KV cache (if present) makes incremental
        #    decoding fast: we don't re-attend over the full prefix each step.
        residual = x
        x = self.input_norm(x)
        x = self.attn(
            x, cos, sin,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            position_ids=position_ids,
        )
        x = residual + x

        # ---- FFN sub-layer ----
        # Same residual + pre-norm shape as the attention sub-layer. The FFN
        # is either:
        #   * a plain SwiGLU (dense layers), or
        #   * an MoE block (most Forge layers) which returns auxiliary
        #     stats the trainer uses for load-balancing.
        residual = x
        x = self.post_attn_norm(x)
        moe_out: Optional[MoEOutput] = None
        if isinstance(self.ffn, MixtureOfExperts):
            moe_out = self.ffn(x)
            x = moe_out.hidden_states
        else:
            x = self.ffn(x)
        x = residual + x

        return x, moe_out
