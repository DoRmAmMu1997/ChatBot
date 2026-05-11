"""Aurora — dense omni-modal decoder-only Transformer (text + image + audio).

Architecture summary:

  * **Vision tower** (SigLIP2-style ViT) encodes images into patch features.
  * **Audio encoder** (Whisper-style Conformer) encodes speech into soft
    tokens.
  * **MLP connectors** project both modalities into the LLM embedding space.
  * The LLM is a stack of ``DecoderBlock``s using GQA + RoPE + SwiGLU.
  * At input time we *splice* image patch embeddings into the text token
    embedding sequence wherever the special ``<|image|>`` token sits, and
    audio embeddings wherever ``<|audio|>`` sits — the standard interleaved
    multimodal pattern used by Llama 3.2-Vision, LLaVA-NeXT, GPT-4o.
  * For audio OUTPUT, the LLM's vocabulary already contains 4096 audio
    code tokens (``<audio:0>``…``<audio:4095>``). When the model emits a
    span between ``<|audio_start|>`` and ``<|audio_end|>``, the inference
    runtime collects those ids and passes them through
    :class:`chatbot.models.audio.AudioCodec` to produce a waveform.

The model is built from scratch using ``torch.nn`` primitives. No third-party
LLM frameworks. The point is for the user (and the user's future readers)
to understand every block.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..audio.encoder import AudioEncoder
from ..common.attention import build_attention
from ..common.ffn import SwiGLU
from ..common.kv_cache import KVCache
from ..common.normalization import RMSNorm
from ..common.rope import build_rotary_embedding
from ..common.transformer import DecoderBlock
from ..vision.connector import MLPConnector
from ..vision.vit_encoder import ViTEncoder
from .config import AuroraConfig


# Token id sentinels — Aurora's tokenizer adds these at training time.
# They are not magic numbers; the model just needs *some* placeholder.
IMAGE_PLACEHOLDER_TOKEN = "<|image|>"
AUDIO_PLACEHOLDER_TOKEN = "<|audio|>"


class AuroraForCausalLM(nn.Module):
    """Aurora: ``LM(text) + ViT(image) + AudioEncoder(speech) → CausalLM``.

    Call signatures:

    1. Pure text:        ``logits = model(input_ids)``.
    2. With images:      ``logits = model(input_ids, images=...)``.
    3. With audio:       ``logits = model(input_ids, audio=...)``.
    4. With both:        ``logits = model(input_ids, images=..., audio=...)``.

    The model finds every position where ``input_ids == image_token_id`` /
    ``audio_token_id`` and replaces that token's embedding with the
    corresponding image's projected patch sequence or audio encoder
    output (one placeholder → many tokens). Downstream layers don't need
    to know anything about modalities — they just see a sequence.
    """

    def __init__(
        self,
        config: AuroraConfig,
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
        # Aurora does NOT use learned absolute position embeddings — RoPE
        # handles position inside attention instead.
        self.rope = build_rotary_embedding(config)

        # ---- Decoder blocks ----
        blocks: List[DecoderBlock] = []
        for _ in range(config.n_layers):
            attn = build_attention(config)
            ffn = SwiGLU(config.d_model, config.ffn_hidden)
            blocks.append(DecoderBlock(d_model=config.d_model, attn=attn, ffn=ffn, rms_eps=config.rms_norm_eps))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # ---- LM head ----
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # ---- Vision tower (optional) ----
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

    # ----- weight init -----

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ----- forward -----

    def _splice_modality(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        modality_embeddings: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Replace placeholder tokens in the text stream with modality embeddings.

        ``modality_embeddings`` is ``[K, T, d_model]`` where K is the number
        of modality items in the batch and T is the per-item token count.
        Callers must pre-allocate exactly T placeholder slots per item in
        ``input_ids``. We do an in-place overwrite of those slots.
        """

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
        """Token + (optional) image / audio embedding to the residual stream."""

        x = self.token_embedding(input_ids)

        # Images: run through ViT + connector, then splice in.
        if images is not None and self.vision_tower is not None and self.image_token_id is not None:
            if images.dim() != 4:
                raise ValueError("`images` must be a 4-D tensor [B*K, 3, H, W]")
            visual = self.vision_connector(self.vision_tower(images))  # [K, T_img, d_model]
            x = self._splice_modality(x, input_ids, visual, self.image_token_id)

        # Audio: run through the audio encoder (which includes mel + projector)
        # and splice in. Each `<|audio|>` placeholder spans the *exact* token
        # count produced by the encoder for that clip.
        if audio is not None and self.audio_encoder is not None and self.audio_token_id is not None:
            if audio.dim() != 2:
                raise ValueError("`audio` must be a 2-D waveform tensor [B*K, samples]")
            audio_tokens = self.audio_encoder(audio)  # [K, T_audio, d_model]
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
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ):
        batch, seq = input_ids.shape

        x = self.embed_inputs(input_ids, images=images, audio=audio)

        cos, sin = self.rope(
            seq_len=seq,
            device=x.device,
            dtype=x.dtype,
            offset=position_offset,
        )

        for layer_idx, block in enumerate(self.blocks):
            x, _moe = block(
                x, cos, sin,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift so logits[t] predicts labels[t+1].
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        eos_token_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Sample tokens autoregressively using a KV cache."""

        from ...inference.sampling import sample_token

        eos_ids = set(eos_token_ids or [self.config.eos_token_id])
        cache = KVCache(num_layers=self.config.n_layers)

        # Prefill — run the full prompt (text + modalities) once, cache everything.
        out = self.forward(input_ids, images=images, audio=audio, kv_cache=cache)
        next_logits = out["logits"][:, -1, :]

        generated = [input_ids]
        for _ in range(max_new_tokens):
            next_token = sample_token(next_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            generated.append(next_token)
            if int(next_token.item()) in eos_ids:
                break
            # Decode step — only the new token, KV cache holds the rest.
            offset = sum(t.shape[1] for t in generated[:-1])
            out = self.forward(next_token, kv_cache=cache, position_offset=offset)
            next_logits = out["logits"][:, -1, :]

        return torch.cat(generated, dim=1)
