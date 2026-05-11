"""Shape and gradient tests for the shared Transformer building blocks."""

from __future__ import annotations

import torch

from chatbot.models.common.attention import GroupedQueryAttention
from chatbot.models.common.ffn import SwiGLU
from chatbot.models.common.normalization import RMSNorm
from chatbot.models.common.rope import RotaryEmbedding, YarnRotaryEmbedding, apply_rotary_emb


def test_rmsnorm_shape_and_grad():
    block = RMSNorm(dim=32)
    x = torch.randn(2, 7, 32, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_swiglu_shape():
    block = SwiGLU(dim=32, hidden=64)
    x = torch.randn(3, 5, 32)
    y = block(x)
    assert y.shape == (3, 5, 32)


def test_rope_shape():
    rope = RotaryEmbedding(head_dim=16, max_position_embeddings=128, base=10000.0)
    cos, sin = rope(seq_len=8, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (8, 16)
    assert sin.shape == (8, 16)


def test_yarn_rope_extends_cache():
    rope = YarnRotaryEmbedding(
        head_dim=16, max_position_embeddings=64, base=10000.0,
        scaling_factor=2.0, original_max_position_embeddings=32,
    )
    cos, sin = rope(seq_len=64, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (64, 16)


def test_gqa_forward():
    attn = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2, head_dim=8)
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=32, base=10000.0)
    x = torch.randn(2, 16, 64)
    cos, sin = rope(seq_len=16, device=x.device, dtype=x.dtype)
    y = attn(x, cos, sin)
    assert y.shape == x.shape


def test_apply_rotary_emb_preserves_shape():
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    cos = torch.randn(8, 16)
    sin = torch.randn(8, 16)
    q2, k2 = apply_rotary_emb(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape
