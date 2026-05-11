"""Strongly-typed config dataclass for Aurora-50B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class _RopeScaling:
    type: str = "none"        # "none" | "linear" | "yarn"
    factor: float = 1.0
    original_max_position_embeddings: int = 4096


@dataclass
class _Vision:
    enabled: bool = True
    image_size: int = 384
    patch_size: int = 14
    vision_dim: int = 1152
    vision_layers: int = 27
    vision_heads: int = 16
    connector_hidden: int = 7168
    num_image_tokens: int = 729


@dataclass
class AuroraConfig:
    """Hyperparameters for the Aurora-50B language stack and vision tower."""

    d_model: int = 7168
    n_layers: int = 64
    n_heads: int = 56
    n_kv_heads: int = 8
    head_dim: int = 128
    ffn_hidden: int = 24576
    vocab_size: int = 131072
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1.0e-5
    rope_base: float = 2000000.0
    rope_scaling: _RopeScaling = field(default_factory=_RopeScaling)
    vision: _Vision = field(default_factory=_Vision)
    dropout: float = 0.0
    init_std: float = 0.02
    tie_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Attention variant. Aurora uses GQA; the dataclass keeps the field so
    # downstream code that branches on it doesn't need to special-case Aurora.
    attention_variant: str = "gqa"


def aurora_config_from_yaml(cfg: DictConfig | Dict[str, Any]) -> AuroraConfig:
    """Build an :class:`AuroraConfig` from a YAML-loaded config block.

    Accepts either an OmegaConf ``DictConfig`` or a plain ``dict``.
    """

    raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    rope_scaling = _RopeScaling(**(raw.get("rope_scaling") or {}))
    vision = _Vision(**(raw.get("vision") or {}))
    attention_variant = "gqa"
    if raw.get("attention"):
        attention_variant = str(raw["attention"].get("variant", "gqa"))
    return AuroraConfig(
        d_model=int(raw["d_model"]),
        n_layers=int(raw["n_layers"]),
        n_heads=int(raw["n_heads"]),
        n_kv_heads=int(raw["n_kv_heads"]),
        head_dim=int(raw.get("head_dim", raw["d_model"] // raw["n_heads"])),
        ffn_hidden=int(raw["ffn_hidden"]),
        vocab_size=int(raw["vocab_size"]),
        max_position_embeddings=int(raw["max_position_embeddings"]),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1.0e-5)),
        rope_base=float(raw.get("rope_base", 10000.0)),
        rope_scaling=rope_scaling,
        vision=vision,
        dropout=float(raw.get("dropout", 0.0)),
        init_std=float(raw.get("init_std", 0.02)),
        tie_embeddings=bool(raw.get("tie_embeddings", False)),
        pad_token_id=int(raw.get("pad_token_id", 0)),
        bos_token_id=int(raw.get("bos_token_id", 1)),
        eos_token_id=int(raw.get("eos_token_id", 2)),
        attention_variant=attention_variant,
    )
