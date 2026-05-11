"""Strongly-typed config dataclass for Aurora.

Aurora is the omni-modal model: text + image + audio in, text + audio out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

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
    connector_hidden: int = 8192
    num_image_tokens: int = 729


@dataclass
class _Audio:
    """Settings for the speech-in encoder and the speech-out codec."""

    enabled: bool = True
    # Encoder (speech-in) settings.
    n_mels: int = 128
    sample_rate: int = 16000
    encoder_dim: int = 1024
    encoder_layers: int = 12
    encoder_heads: int = 16
    # Codec (speech-out) settings — must match the audio codec model.
    codec_sample_rate: int = 24000
    num_audio_codes: int = 4096


@dataclass
class AuroraConfig:
    """Hyperparameters for the Aurora omni-modal LLM, vision tower and audio I/O."""

    d_model: int = 8192
    n_layers: int = 72
    n_heads: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128
    ffn_hidden: int = 28672
    # The vocab includes text + 4096 audio code tokens + audio/image specials.
    # Default 132096 = 128K text BPE + 4096 audio + ~64 spec slots rounded.
    vocab_size: int = 132096
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1.0e-5
    rope_base: float = 2000000.0
    rope_scaling: _RopeScaling = field(default_factory=_RopeScaling)
    vision: _Vision = field(default_factory=_Vision)
    audio: _Audio = field(default_factory=_Audio)
    dropout: float = 0.0
    init_std: float = 0.02
    tie_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    attention_variant: str = "gqa"


def aurora_config_from_yaml(cfg: DictConfig | Dict[str, Any]) -> AuroraConfig:
    """Build an :class:`AuroraConfig` from a YAML-loaded config block."""

    raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    rope_scaling = _RopeScaling(**(raw.get("rope_scaling") or {}))
    vision = _Vision(**(raw.get("vision") or {}))
    audio = _Audio(**(raw.get("audio") or {}))
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
        audio=audio,
        dropout=float(raw.get("dropout", 0.0)),
        init_std=float(raw.get("init_std", 0.02)),
        tie_embeddings=bool(raw.get("tie_embeddings", False)),
        pad_token_id=int(raw.get("pad_token_id", 0)),
        bos_token_id=int(raw.get("bos_token_id", 1)),
        eos_token_id=int(raw.get("eos_token_id", 2)),
        attention_variant=attention_variant,
    )
