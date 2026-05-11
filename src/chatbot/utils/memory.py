"""Parameter and memory accounting helpers.

These functions are pure-Python — they introspect the model graph and don't
need a forward pass. We use them in ``scripts/count_params.py`` to verify
that a config really hits ~50B (Aurora) or ~250B total / ~25B active (Forge)
*before* anyone burns money on GPUs.
"""

from __future__ import annotations

from typing import Tuple


def count_parameters(model) -> Tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)`` for a torch module."""

    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def count_active_parameters(model) -> int:
    """Approximate number of parameters touched by a single forward token.

    For a dense model this equals ``count_parameters(model)[0]``. For an MoE
    model it's the embeddings + non-expert weights + (k * per-expert weights)
    where k is the number of active experts per token.

    We dedupe by tensor identity so tied embeddings (where ``lm_head.weight``
    points at ``token_embedding.weight``) are counted exactly once.
    """

    seen_param_ids = set()
    total = 0

    def _add_unique(params_iter):
        nonlocal total
        for p in params_iter:
            if id(p) in seen_param_ids:
                continue
            seen_param_ids.add(id(p))
            total += p.numel()

    # First pass: find every MoE block and account for its experts at the
    # active-rate. Record their IDs so we can skip them in the dense pass.
    moe_module_ids = set()
    expert_param_ids = set()
    for module in model.modules():
        if hasattr(module, "experts") and hasattr(module, "num_active_experts"):
            moe_module_ids.add(id(module))
            num_routed = int(getattr(module, "num_routed_experts", 0))
            num_active = int(module.num_active_experts)
            for p in module.experts.parameters():
                expert_param_ids.add(id(p))
            if num_routed > 0:
                per_expert = sum(p.numel() for p in module.experts.parameters())
                # Average expert FLOPs per token = active_share × total expert params.
                total += int(per_expert * num_active / num_routed)

    # Second pass: count every parameter that ISN'T inside an expert pool.
    # The walk over ``model.parameters()`` naturally deduplicates tied
    # tensors because PyTorch returns each parameter only once.
    for p in model.parameters():
        if id(p) in expert_param_ids:
            continue
        if id(p) in seen_param_ids:
            continue
        seen_param_ids.add(id(p))
        total += p.numel()

    return total


def format_param_count(n: int) -> str:
    """Human-readable string for a parameter count: 49_000_000_000 → '49.0B'."""

    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)
