"""Pretraining entrypoint (next-token prediction).

Shape of a training step:

    1. Pull a batch of packed token sequences from the data loader.
    2. Forward the model — get loss + (optional) MoE aux losses.
    3. Backward, clip gradients, optimizer step, scheduler step.
    4. Periodically log; periodically save.

This module focuses on the *logic*. Distributed wrapping happens through
``training.distributed`` and is opt-in via the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..data.pretrain_loader import MixEntry, PretrainDataset, collate_pretrain_batch
from ..models.aurora_50b import AuroraForCausalLM, aurora_config_from_yaml
from ..models.forge_250b import ForgeForCausalLM, forge_config_from_yaml
from ..tokenizer.bpe import BPETokenizer
from ..utils.config import load_config, override_from_cli, save_config
from ..utils.logging import get_logger, setup_logging
from ..utils.seeding import set_seed
from .checkpoint import save_checkpoint
from .distributed import init_distributed, wrap_fsdp
from .metrics import RollingMean
from .optim import build_optimizer, build_scheduler

logger = get_logger(__name__)


def build_model(model_cfg: DictConfig) -> torch.nn.Module:
    family = str(model_cfg.get("family", "aurora")).lower()
    if family == "aurora":
        cfg = aurora_config_from_yaml(model_cfg)
        return AuroraForCausalLM(cfg)
    if family == "forge":
        cfg = forge_config_from_yaml(model_cfg)
        return ForgeForCausalLM(cfg)
    raise ValueError(f"Unknown model family: {family!r}")


def run_pretrain(
    *,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    tokenizer_path: str,
    overrides: Any = None,
) -> Path:
    setup_logging(level="INFO")
    set_seed(int(train_cfg.get("seed", 42)))
    dist_env = init_distributed(str(train_cfg.distributed.get("backend", "none")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Building model on rank %d / %d", dist_env.rank, dist_env.world_size)
    model = build_model(model_cfg)
    model.to(device)

    if dist_env.backend == "fsdp":
        model = wrap_fsdp(model, cfg=train_cfg.distributed.get("fsdp", {}))

    tokenizer = BPETokenizer.from_file(tokenizer_path)

    mix = [MixEntry(name=str(e["name"]), weight=float(e["weight"])) for e in train_cfg.data.mix]
    dataset = PretrainDataset(
        mix=mix,
        tokenizer=tokenizer,
        block_size=int(train_cfg.sequence_length),
        eos_token_id=tokenizer.eos_id(),
        pad_token_id=tokenizer.pad_id(),
        shuffle_buffer=int(train_cfg.data.get("shuffle_buffer", 100000)),
        seed=int(train_cfg.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.micro_batch_size),
        collate_fn=collate_pretrain_batch,
        num_workers=int(train_cfg.data.get("num_workers", 0)),
    )

    optimizer = build_optimizer(model.parameters(), train_cfg.optimizer)
    scheduler = build_scheduler(
        optimizer, train_cfg.scheduler, total_steps=int(train_cfg.max_steps)
    )

    output_dir = Path(train_cfg.get("output_dir", "outputs/pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(model_cfg, output_dir / "model_config.yaml")
    save_config(train_cfg, output_dir / "train_config.yaml")

    rolling = RollingMean(window=int(train_cfg.get("log_every", 25)) * 4)

    model.train()
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    grad_accum = int(train_cfg.get("grad_accumulation", 1))

    for batch in loader:
        if step >= int(train_cfg.max_steps):
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass.
        # ``input_ids`` is a [batch, seq] tensor of token ids; ``labels`` is
        # the same shape, shifted left by one so position t's label is the
        # token that *should* come after position t. The model returns the
        # cross-entropy loss for us (averaged over non-pad tokens).
        out = model(input_ids=input_ids, labels=labels)
        # Gradient accumulation: if we want a global batch larger than fits
        # on a single GPU, we accumulate grads over ``grad_accum`` mini-batches
        # before stepping the optimizer. Dividing the loss here keeps the
        # effective LR identical regardless of the accumulation factor.
        loss = out["loss"] / grad_accum
        # Backward pass — populate ``.grad`` on every trainable parameter.
        loss.backward()

        accum_loss += float(loss.item()) * grad_accum
        if (step + 1) % grad_accum == 0:
            # Clip gradients to avoid a single bad batch (e.g. unusual token
            # mix) from blowing up the loss. Standard practice at scale.
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip))
            optimizer.step()       # apply the gradient update
            scheduler.step()       # advance the LR schedule (cosine-with-warmup)
            optimizer.zero_grad(set_to_none=True)  # reset for the next accumulation cycle

        rolling.update(accum_loss)
        accum_loss = 0.0

        step += 1
        if step % int(train_cfg.get("log_every", 25)) == 0 and dist_env.is_main:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                "step %d / %d | loss %.4f | lr %.2e",
                step,
                int(train_cfg.max_steps),
                rolling.mean,
                lr,
            )

        if step % int(train_cfg.get("save_every", 1000)) == 0 and dist_env.is_main:
            save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=model_cfg,
            )

    if dist_env.is_main:
        save_checkpoint(
            output_dir=output_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=model_cfg,
        )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain Aurora or Forge.")
    parser.add_argument("--model", required=True, help="Model config name, e.g. 'tiny' or 'aurora-50b'.")
    parser.add_argument("--training", default="pretrain", help="Training config name (under configs/training).")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    parser.add_argument("override", nargs="*", help="Dot-list overrides, e.g. training.lr=5e-5.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_cfg = load_config(f"models/{args.model}")
    train_cfg = load_config(f"training/{args.training}")
    train_cfg = override_from_cli(train_cfg, args.override)
    run_pretrain(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer_path=args.tokenizer,
    )
