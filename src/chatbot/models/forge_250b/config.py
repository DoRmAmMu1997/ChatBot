"""Strongly-typed config dataclass for Forge-250B."""

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
    num_routed_experts: int = 128
    num_shared_experts: int = 1
    num_active_experts: int = 8
    expert_hidden: int = 1536
    router_jitter: float = 0.0
    load_balancing: str = "aux_loss_free"
    bias_update_speed: float = 1.0e-3
    router_z_loss_coef: float = 1.0e-4
    aux_loss_coef: float = 0.0


@dataclass
class _FFN:
    type: str = "swiglu"
    hidden: int = 18432
    num_dense_layers: int = 3


@dataclass
class ForgeConfig:
    """Hyperparameters for the Forge-250B language stack."""

    d_model: int = 6144
    n_layers: int = 64
    n_heads: int = 48
    n_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 200032
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1.0e-6
    rope_base: float = 50000000.0
    rope_scaling: _RopeScaling = field(default_factory=_RopeScaling)
    attention: _Attention = field(default_factory=_Attention)
    moe: _MoE = field(default_factory=_MoE)
    ffn: _FFN = field(default_factory=_FFN)
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
        dropout=float(raw.get("dropout", 0.0)),
        init_std=float(raw.get("init_std", 0.02)),
        tie_embeddings=bool(raw.get("tie_embeddings", False)),
        pad_token_id=int(raw.get("pad_token_id", 0)),
        bos_token_id=int(raw.get("bos_token_id", 1)),
        eos_token_id=int(raw.get("eos_token_id", 2)),
    )
