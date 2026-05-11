"""Shared Transformer building blocks (RMSNorm, RoPE, GQA, MLA, SwiGLU, MoE).

These are imported by both Aurora-50B (dense + GQA + vision) and Forge-250B
(MoE + MLA). Keeping them in one place means a fix in attention.py or a
correctness improvement in moe.py automatically benefits both models.
"""

from .attention import GroupedQueryAttention, MultiHeadLatentAttention, build_attention
from .ffn import SwiGLU
from .kv_cache import KVCache, MLAKVCache
from .moe import MixtureOfExperts
from .normalization import RMSNorm
from .rope import RotaryEmbedding, YarnRotaryEmbedding, apply_rotary_emb
from .transformer import DecoderBlock

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "YarnRotaryEmbedding",
    "apply_rotary_emb",
    "GroupedQueryAttention",
    "MultiHeadLatentAttention",
    "build_attention",
    "SwiGLU",
    "MixtureOfExperts",
    "KVCache",
    "MLAKVCache",
    "DecoderBlock",
]
