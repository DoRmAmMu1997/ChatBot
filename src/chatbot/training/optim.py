"""Optimizer and LR-schedule factories.

We support a single optimizer (AdamW) and the standard cosine-with-warmup
schedule. Adding new ones is intentionally a small change — see the
``build_optimizer`` switch.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg) -> Optimizer:
    """Return the configured optimizer.

    The config block looks like::

        optimizer:
          name: adamw
          lr: 3.0e-4
          weight_decay: 0.1
          betas: [0.9, 0.95]
          eps: 1.0e-8
    """

    name = str(cfg.get("name", "adamw")).lower()
    if name == "adamw":
        return AdamW(
            parameters,
            lr=float(cfg.lr),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            betas=tuple(cfg.get("betas", [0.9, 0.95])),
            eps=float(cfg.get("eps", 1.0e-8)),
        )
    raise ValueError(f"Unknown optimizer name: {name!r}")


def build_scheduler(optimizer: Optimizer, cfg, total_steps: int) -> LambdaLR:
    """Linear warmup → cosine decay to ``min_lr``."""

    name = str(cfg.get("name", "cosine_with_warmup")).lower()
    if name != "cosine_with_warmup":
        raise ValueError(f"Unknown scheduler: {name!r}")
    warmup = int(cfg.get("warmup_steps", 0))
    base_lr = float(optimizer.defaults["lr"])
    min_lr = float(cfg.get("min_lr", base_lr * 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            # Linear warmup from 0 → 1 over ``warmup`` steps.
            return float(step + 1) / float(max(warmup, 1))
        # Cosine decay from 1 → min_lr / base_lr.
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        floor = min_lr / base_lr
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
