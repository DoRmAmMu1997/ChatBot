"""Checkpoint save / load. Sharded for big models, simple for tiny.

For full-fine-tune training we save the model state dict plus optimizer
state plus the resolved config + a step counter. ``torch.save`` handles
sharding of FSDP-flattened parameters when you collect them with
``FullStateDictConfig``. For tiny configs we just dump everything in one
file because the whole thing fits in memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    config: Optional[DictConfig] = None,
    extras: Optional[dict] = None,
) -> Path:
    """Save a checkpoint and return its directory."""

    ckpt_dir = Path(output_dir) / f"step_{step:09d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    if config is not None:
        with open(ckpt_dir / "config.yaml", "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(config, resolve=True))
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "extras": extras or {}}, f, indent=2)

    # Update / refresh the ``latest`` pointer (a small file, not a symlink, so
    # it works on Windows too).
    with open(Path(output_dir) / "latest", "w", encoding="utf-8") as f:
        f.write(str(ckpt_dir))

    return ckpt_dir


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: str = "cpu",
) -> int:
    """Load model (and optionally optimizer / scheduler) state. Returns step."""

    path = Path(path)
    if path.is_file():
        # Plain ``.pt`` file path — assume it's just the model state.
        state = torch.load(path, map_location=map_location)
        model.load_state_dict(state)
        return 0

    if (path / "latest").exists():
        # If ``path`` is an output dir, follow the ``latest`` pointer.
        with open(path / "latest", "r", encoding="utf-8") as f:
            path = Path(f.read().strip())

    state = torch.load(path / "model.pt", map_location=map_location)
    model.load_state_dict(state)
    if optimizer is not None and (path / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=map_location))
    if scheduler is not None and (path / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location=map_location))
    meta_path = path / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("step", 0))
    return 0
