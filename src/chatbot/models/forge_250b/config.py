"""Strongly-typed config dataclass for Forge.

v2 adds:

* Vision tower (smaller than Aurora's — code screenshots not fine art).
* Audio encoder (lighter than Aurora's — short voice descriptions).
* Bumped MoE size: 160 routed experts × 2048 expert-hidden (was 128 × 1536).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


@dataclass
class _RopeScaling:
    type: str = "none"
    factor: float = 1.0
    original_max_position_embeddings: int = 4096


@dataclass
class _Attention:
    variant: str = "mla"
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128


@dataclass
class _MoE:
    enabled: bool = True
    num_routed_experts: int = 160
    num_shared_experts: int = 1
    num_active_experts: int = 8
    expert_hidden: int = 2048
    router_jitter: float = 0.0
    load_balancing: str = "aux_loss_free"
    bias_update_speed: float = 1.0e-3
    router_z_loss_coef: float = 1.0e-4
    aux_loss_coef: float = 0.0


@dataclass
class _FFN:
    type: str = "swiglu"
    hidden: int = 19968
    num_dense_layers: int = 3


@dataclass
class _Vision:
    """Vision tower for Forge — sized for code screenshots, not natural images."""

    enabled: bool = True
    image_size: int = 224
    patch_size: int = 14
    vision_dim: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    connector_hidden: int = 6656
    num_image_tokens: int = 256        # 16x16 grid at 224/14 = 16 (so 256 patch tokens)


@dataclass
class _Audio:
    """Audio I/O for Forge — voice descriptions, short clips."""

    enabled: bool = True
    n_mels: int = 128
    sample_rate: int = 16000
    encoder_dim: int = 1024
    encoder_layers: int = 8            # lighter than Aurora's 12
    encoder_heads: int = 16
    codec_sample_rate: int = 24000
    num_audio_codes: int = 4096


@dataclass
class ForgeConfig:
    """Hyperparameters for the Forge MoE coder LLM, vision tower, audio I/O."""

    d_model: int = 6656
    n_layers: int = 72
    n_heads: int = 52
    n_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 204096          # 200K text BPE + 4096 audio codes
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1.0e-6
    rope_base: float = 50000000.0
    rope_scaling: _RopeScaling = field(default_factory=_RopeScaling)
    attention: _Attention = field(default_factory=_Attention)
    moe: _MoE = field(default_factory=_MoE)
    ffn: _FFN = field(default_factory=_FFN)
    vision: _Vision = field(default_factory=_Vision)
    audio: _Audio = field(default_factory=_Audio)
    dropout: float = 0.0
    init_std: float = 0.02
    tie_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def attention_variant(self) -> str:
        return self.attention.variant


def forge_config_from_yaml(cfg: DictConfig | Dict[str, Any]) -> ForgeConfig:
    raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    return ForgeConfig(
        d_model=int(raw["d_model"]),
        n_layers=int(raw["n_layers"]),
        n_heads=int(raw["n_heads"]),
        n_kv_heads=int(raw["n_kv_heads"]),
        head_dim=int(raw.get("head_dim", raw["d_model"] // raw["n_heads"])),
        vocab_size=int(raw["vocab_size"]),
        max_position_embeddings=int(raw["max_position_embeddings"]),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1.0e-6)),
        rope_base=float(raw.get("rope_base", 50000000.0)),
        rope_scaling=_RopeScaling(**(raw.get("rope_scaling") or {})),
        attention=_Attention(**(raw.get("attention") or {})),
        moe=_MoE(**(raw.get("moe") or {})),
        ffn=_FFN(**(raw.get("ffn") or {})),
        vision=_Vision(**(raw.get("vision") or {})),
        audio=_Audio(**(raw.get("audio") or {})),
        dropout=float(raw.get("dropout", 0.0)),
        init_std=float(raw.get("init_std", 0.02)),
        tie_embeddings=bool(raw.get("tie_embeddings", False)),
        pad_token_id=int(raw.get("pad_token_id", 0)),
        bos_token_id=int(raw.get("bos_token_id", 1)),
        eos_token_id=int(raw.get("eos_token_id", 2)),
    )
