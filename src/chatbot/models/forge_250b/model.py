"""Forge — Mixture-of-Experts coder with MLA, vision, and audio I/O.

Architecture summary:

  * Token embeddings → stack of decoder blocks → RMSNorm → LM head.
  * Attention is **MLA** (multi-head latent attention, DeepSeek-V3-style) —
    the KV cache stores a compressed latent so 1M-context inference is
    feasible in memory.
  * The first ``ffn.num_dense_layers`` blocks use dense SwiGLU FFNs as a
    "warm-up" — DeepSeek-V3 found this stabilizes early training. The
    remaining blocks use the fine-grained MoE FFN.
  * **Vision tower** (small SigLIP2-style ViT, tuned for code screenshots)
    + MLP connector splice image patches into the residual stream where
    ``<|image|>`` placeholders sit.
  * **Audio encoder** (Whisper-style Conformer, 8 layers) splices
    voice-description tokens in at ``<|audio|>`` placeholders.
  * **Audio output**: like Aurora, the LLM's vocabulary contains 4096
    audio code tokens that the runtime decodes through the shared codec.

The MoE aux losses (``aux_loss`` + ``router_z_loss``) are surfaced through
the forward output so the training loop can sum them into the LM loss.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.encoder import AudioEncoder
from ..common.attention import build_attention
from ..common.ffn import SwiGLU
from ..common.kv_cache import KVCache, MLAKVCache
from ..common.moe import MixtureOfExperts
from ..common.normalization import RMSNorm
from ..common.rope import build_rotary_embedding
from ..common.transformer import DecoderBlock
from ..vision.connector import MLPConnector
from ..vision.vit_encoder import ViTEncoder
from .config import ForgeConfig


IMAGE_PLACEHOLDER_TOKEN = "<|image|>"
AUDIO_PLACEHOLDER_TOKEN = "<|audio|>"


class ForgeForCausalLM(nn.Module):
    """Forge MoE coder LLM with optional vision + audio inputs."""

    def __init__(
        self,
        config: ForgeConfig,
        *,
        image_token_id: Optional[int] = None,
        audio_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id

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
                # Remaining layers: MoE — 160 small experts, top-8 routing,
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

        # ---- Vision tower (optional) ----
        # Code screenshots, not natural images, so we size smaller than
        # Aurora's tower: 224 px, 12 layers, 768-d.
        if config.vision.enabled:
            self.vision_tower = ViTEncoder(
                image_size=config.vision.image_size,
                patch_size=config.vision.patch_size,
                dim=config.vision.vision_dim,
                depth=config.vision.vision_layers,
                num_heads=config.vision.vision_heads,
            )
            self.vision_connector = MLPConnector(
                vision_dim=config.vision.vision_dim,
                hidden_dim=config.vision.connector_hidden,
                llm_dim=config.d_model,
                pool=None,
            )
        else:
            self.vision_tower = None
            self.vision_connector = None

        # ---- Audio encoder (optional) ----
        # Lighter than Aurora's: 8 layers instead of 12 because Forge's
        # audio use case (voice descriptions of bugs) is shorter and less
        # demanding than open-domain conversation.
        if config.audio.enabled:
            self.audio_encoder = AudioEncoder(
                llm_dim=config.d_model,
                n_mels=config.audio.n_mels,
                dim=config.audio.encoder_dim,
                depth=config.audio.encoder_layers,
                num_heads=config.audio.encoder_heads,
                sample_rate=config.audio.sample_rate,
            )
        else:
            self.audio_encoder = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _splice_modality(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        modality_embeddings: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Same in-place splicing logic Aurora uses; see its docstring for detail."""

        positions = (input_ids == placeholder_token_id).nonzero(as_tuple=False)
        if positions.numel() == 0:
            return x
        per_item = modality_embeddings.shape[1]
        flat = modality_embeddings.reshape(-1, x.shape[-1])
        seen = 0
        for b, t in positions.tolist():
            end = t + per_item
            if end > x.shape[1]:
                raise ValueError(
                    f"Not enough placeholder slots for modality token {placeholder_token_id}."
                )
            x[b, t:end, :] = flat[seen : seen + per_item]
            seen += per_item
        return x

    def embed_inputs(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        if images is not None and self.vision_tower is not None and self.image_token_id is not None:
            if images.dim() != 4:
                raise ValueError("`images` must be a 4-D tensor [B*K, 3, H, W]")
            visual = self.vision_connector(self.vision_tower(images))
            x = self._splice_modality(x, input_ids, visual, self.image_token_id)
        if audio is not None and self.audio_encoder is not None and self.audio_token_id is not None:
            if audio.dim() != 2:
                raise ValueError("`audio` must be a 2-D waveform tensor [B*K, samples]")
            audio_tokens = self.audio_encoder(audio)
            x = self._splice_modality(x, input_ids, audio_tokens, self.audio_token_id)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[MLAKVCache] = None,
        position_offset: int = 0,
    ):
        batch, seq = input_ids.shape

        x = self.embed_inputs(input_ids, images=images, audio=audio)
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
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 0,
        eos_token_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        from ...inference.sampling import sample_token

        eos_ids = set(eos_token_ids or [self.config.eos_token_id])
        cache = MLAKVCache(num_layers=self.config.n_layers)

        out = self.forward(input_ids, images=images, audio=audio, kv_cache=cache)
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
