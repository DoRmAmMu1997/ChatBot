"""Thin wrappers around FSDP2 and DeepSpeed-ZeRO so the trainers stay tidy.

If you're new to distributed training:

* **FSDP** (Fully Sharded Data Parallel) splits a model's parameters, gradients,
  and optimizer state across multiple GPUs. Each GPU only ever materializes
  the shard it owns plus the layer it's currently computing — this is what
  lets you train a 50B-parameter model on hardware that can't hold all 50B
  on one card.
* **DeepSpeed ZeRO-3** does the same thing with a different code path; it's
  more battle-tested at the 250B-class size for some workloads, so we let
  you switch via config.
* **None** mode skips both and runs single-process — useful for the tiny
  smoke configs and for LoRA fine-tuning on a single GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DistEnv:
    """Lightweight handle to the current distributed environment."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_main: bool = True
    backend: str = "none"


def init_distributed(backend: str = "none") -> DistEnv:
    """Initialize torch.distributed if needed and return a description.

    ``backend`` is one of ``"none"`` | ``"fsdp"`` | ``"deepspeed"``. The
    first two use ``torch.distributed`` directly; the third defers to
    ``deepspeed.init_distributed`` and we just read the env vars.
    """

    if backend == "none":
        return DistEnv(rank=0, world_size=1, local_rank=0, is_main=True, backend="none")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if backend == "fsdp":
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
    elif backend == "deepspeed":
        try:
            import deepspeed

            deepspeed.init_distributed(dist_backend="nccl" if torch.cuda.is_available() else "gloo")
        except ImportError as exc:
            raise RuntimeError(
                "DeepSpeed backend requested but the 'deepspeed' package is not installed."
            ) from exc
    else:
        raise ValueError(f"Unknown distributed backend: {backend!r}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=rank == 0,
        backend=backend,
    )


def wrap_fsdp(model: nn.Module, *, cfg) -> nn.Module:
    """Wrap a model with FSDP2 according to a config block.

    The auto-wrap policy uses a size threshold — every leaf module with
    > 1M parameters becomes its own FSDP unit. This balances bookkeeping
    cost with sharding granularity for our typical model sizes.
    """

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    import functools

    strategy_name = str(cfg.get("sharding_strategy", "FULL_SHARD"))
    strategy = getattr(ShardingStrategy, strategy_name)
    cpu_offload = bool(cfg.get("cpu_offload", False))
    use_orig_params = bool(cfg.get("use_orig_params", True))

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000,
    )

    return FSDP(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=use_orig_params,
        cpu_offload=None if not cpu_offload else _make_cpu_offload(),
    )


def _make_cpu_offload():
    from torch.distributed.fsdp import CPUOffload

    return CPUOffload(offload_params=True)


def wrap_deepspeed(model: nn.Module, *, optimizer, cfg) -> tuple:
    """Hand a model to DeepSpeed. Returns ``(engine, optimizer)``."""

    import deepspeed

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,  # caller overrides via launcher args
        "zero_optimization": {
            "stage": int(cfg.get("zero_stage", 3)),
            "offload_optimizer": (
                {"device": "cpu", "pin_memory": True} if cfg.get("offload_optimizer") else None
            ),
            "offload_param": (
                {"device": "cpu", "pin_memory": True} if cfg.get("offload_param") else None
            ),
        },
        "bf16": {"enabled": True},
    }
    engine, ds_opt, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
    return engine, ds_opt
