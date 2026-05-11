"""LoRA — Low-Rank Adaptation of large language models.

The idea in one paragraph:

A frozen ``Linear(in, out)`` is a big matrix ``W`` (e.g. 7168 × 7168 = 51M
parameters). Instead of training the whole thing, LoRA freezes ``W`` and
adds two tiny matrices ``A`` (in × r) and ``B`` (r × out) where ``r`` is
small (e.g. 16-64). The new forward is::

    y = (W + α/r · B A) x

Only ``A`` and ``B`` get gradients — the original ``W`` is frozen. For
``r = 32``, the trainable share drops from 51M to ~460K (a 100× reduction).
The base model can be left in 4-bit (QLoRA) or bf16, and only the adapters
are kept in higher precision.

This module ships a plain ``LoRALinear`` that wraps ``nn.Linear`` and a
recursive ``apply_lora_to_model`` walker that swaps targeted layers.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """``Linear`` plus a trainable low-rank adapter.

    Args:
        base_linear: the frozen original layer.
        rank: ``r`` — width of the low-rank bottleneck.
        alpha: scaling factor; effective scale is ``alpha / rank``.
        dropout: dropout applied to the adapter input.
        init: ``"kaiming"`` (Kaiming-uniform for A, zeros for B — standard)
            or ``"zero"`` (both zero, makes the adapter a no-op at init).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        init: str = "kaiming",
    ):
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.base = base_linear
        # The base weights stay frozen — only A and B train.
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        # ``A`` projects down to rank-dimensional space; ``B`` projects back up.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        if init == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif init == "zero":
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
        else:
            raise ValueError(f"Unknown LoRA init: {init!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path — same output as the original Linear.
        base_out = self.base(x)
        # Adapter path — x @ A^T @ B^T (rank-bottlenecked).
        x_dropped = self.dropout(x)
        adapter = (x_dropped @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return base_out + adapter * self.scale

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha_over_rank={self.scale:.3f}"


def apply_lora_to_model(
    model: nn.Module,
    *,
    target_modules: Iterable[str],
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.0,
    init: str = "kaiming",
) -> nn.Module:
    """Walk the model and replace every ``Linear`` whose name matches a target.

    ``target_modules`` is a list of short names (e.g. ``"q_proj"``,
    ``"gate_proj"``). Any module whose qualified name ENDS with one of those
    strings gets wrapped in :class:`LoRALinear`. This matches the naming
    convention used in our Transformer blocks.
    """

    targets = list(target_modules)

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(qualified.endswith(t) for t in targets):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout, init=init)
                setattr(parent, name, wrapped)
            else:
                _replace(child, qualified)

    # Freeze everything by default; LoRALinear will re-enable just its own
    # adapter parameters during ``_replace``.
    for param in model.parameters():
        param.requires_grad = False

    _replace(model)

    # Re-enable adapter params explicitly (they default to ``requires_grad=True``
    # but we just turned all params off above).
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    return model


def save_lora_adapters(model: nn.Module, output_dir: str) -> None:
    """Save only the LoRA adapter weights (much smaller than full model)."""

    os.makedirs(output_dir, exist_ok=True)
    state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    torch.save(state, os.path.join(output_dir, "adapters.pt"))


def load_lora_adapters(model: nn.Module, path: str) -> nn.Module:
    """Load saved adapter weights into a model that already has LoRA wrappers."""

    state = torch.load(path, map_location="cpu")
    lookup = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    for key, tensor in state.items():
        module_name, _, param_name = key.rpartition(".")
        target = lookup.get(module_name)
        if target is None:
            continue
        getattr(target, param_name).data.copy_(tensor)
    return model
