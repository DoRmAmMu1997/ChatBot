"""Forge-250B model — Mixture-of-Experts coder with Multi-head Latent Attention.

Architecture summary:

  * Token embeddings → stack of decoder blocks → RMSNorm → LM head.
  * Attention is **MLA** (multi-head latent attention, DeepSeek-V3-style) —
    the KV cache stores a compressed latent so 1M-context inference is
    feasible in memory.
  * The first ``ffn.num_dense_layers`` blocks use dense SwiGLU FFNs as a
    "warm-up" — DeepSeek-V3 found this stabilizes early training. The
    remaining blocks use the fine-grained MoE FFN.
  * No vision tower. Forge is a code/SWE specialist.

The MoE aux losses (``aux_loss`` + ``router_z_loss``) are surfaced through
the forward output so the training loop can sum them into the LM loss.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.attention import build_attention
from ..common.ffn import SwiGLU
from ..common.kv_cache import KVCache, MLAKVCache
from ..common.moe import MixtureOfExperts
from ..common.normalization import RMSNorm
from ..common.rope import build_rotary_embedding
from ..common.transformer import DecoderBlock
from .config import ForgeConfig


class ForgeForCausalLM(nn.Module):
    """Forge-250B MoE coder LLM."""

    def __init__(self, config: ForgeConfig):
        super().__init__()
        self.config = config

        # ---- Embeddings ----
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = build_rotary_embedding(config)

        # ---- Decoder blocks ----
        # DeepSeek-V3 found that using *dense* SwiGLU FFNs for the first few
        # layers (then switching to MoE for the rest) stabilizes early
        # training. Intuition: the router has to learn how to route tokens
        # *to experts that haven't learned anything yet*, so early routing
        # decisions are random; dense layers buy a few stable epochs before
        # the experts come online.
        blocks: List[DecoderBlock] = []
        num_dense = int(config.ffn.num_dense_layers)
        for layer_idx in range(config.n_layers):
            attn = build_attention(config)
            if layer_idx < num_dense:
                # First few layers: plain SwiGLU FFN.
                ffn = SwiGLU(config.d_model, config.ffn.hidden)
            else:
                # Remaining layers: MoE — 128 small experts, top-8 routing,
                # plus one shared expert that's always active.
                ffn = MixtureOfExperts(
                    d_model=config.d_model,
                    num_routed_experts=config.moe.num_routed_experts,
                    num_shared_experts=config.moe.num_shared_experts,
                    num_active_experts=config.moe.num_active_experts,
                    expert_hidden=config.moe.expert_hidden,
                    router_jitter=config.moe.router_jitter,
                    load_balancing=config.moe.load_balancing,
                    bias_update_speed=config.moe.bias_update_speed,
                    router_z_loss_coef=config.moe.router_z_loss_coef,
                    aux_loss_coef=config.moe.aux_loss_coef,
                )
            blocks.append(DecoderBlock(d_model=config.d_model, attn=attn, ffn=ffn, rms_eps=config.rms_norm_eps))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # ---- LM head ----
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[MLAKVCache] = None,
        position_offset: int = 0,
    ):
        batch, seq = input_ids.shape

        x = self.token_embedding(input_ids)
        cos, sin = self.rope(seq_len=seq, device=x.device, dtype=x.dtype, offset=position_offset)

        total_aux = x.new_zeros(())
        total_router_z = x.new_zeros(())
        for layer_idx, block in enumerate(self.blocks):
            x, moe_out = block(
                x, cos, sin,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )
            if moe_out is not None:
                total_aux = total_aux + moe_out.aux_loss
                total_router_z = total_router_z + moe_out.router_z_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
            loss = lm_loss + total_aux + total_router_z
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux,
            "router_z_loss": total_router_z,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 0,
        eos_token_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        from ...inference.sampling import sample_token

        eos_ids = set(eos_token_ids or [self.config.eos_token_id])
        cache = MLAKVCache(num_layers=self.config.n_layers)

        out = self.forward(input_ids, kv_cache=cache)
        next_logits = out["logits"][:, -1, :]

        generated = [input_ids]
        for _ in range(max_new_tokens):
            next_token = sample_token(next_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            generated.append(next_token)
            if int(next_token.item()) in eos_ids:
                break
            offset = sum(t.shape[1] for t in generated[:-1])
            out = self.forward(next_token, kv_cache=cache, position_offset=offset)
            next_logits = out["logits"][:, -1, :]

        return torch.cat(generated, dim=1)
