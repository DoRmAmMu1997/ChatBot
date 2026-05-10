"""Parameter-count utilities that do not allocate model weights."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig


@dataclass(frozen=True)
class ParameterReport:
    """Human-readable parameter count summary."""

    total: int
    embeddings: int
    layers: int
    final_norm: int
    lm_head: int


def estimate_parameter_count(config: ModelConfig) -> ParameterReport:
    """Estimate parameter count without importing PyTorch."""

    head_dim = config.head_dim
    kv_dim = config.n_kv_head * head_dim
    embeddings = config.vocab_size * config.n_embd

    q_proj = config.n_embd * (config.n_head * head_dim)
    k_proj = config.n_embd * kv_dim
    v_proj = config.n_embd * kv_dim
    o_proj = config.n_embd * config.n_embd
    attention = q_proj + k_proj + v_proj + o_proj

    mlp = 3 * config.n_embd * config.ffn_hidden_size
    norms = 2 * config.n_embd
    layers = config.n_layer * (attention + mlp + norms)
    final_norm = config.n_embd
    lm_head = 0 if config.tie_embeddings else config.vocab_size * config.n_embd

    return ParameterReport(
        total=embeddings + layers + final_norm + lm_head,
        embeddings=embeddings,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )
