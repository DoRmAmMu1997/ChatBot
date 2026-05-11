"""Tool-use SFT loop. Mirrors the plain SFT loop but uses the tool-use loader."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..data.tool_use_loader import ToolMixEntry, ToolUseDataset
from ..data.sft_loader import collate_sft_batch
from ..tokenizer.bpe import BPETokenizer
from ..utils.config import load_config, override_from_cli, save_config
from ..utils.logging import get_logger, setup_logging
from ..utils.seeding import set_seed
from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import init_distributed, wrap_fsdp
from .metrics import RollingMean
from .optim import build_optimizer, build_scheduler
from .pretrain import build_model

logger = get_logger(__name__)


def run_tool_use_sft(
    *,
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    tokenizer_path: str,
    resume_from: str,
) -> Path:
    setup_logging(level="INFO")
    set_seed(int(train_cfg.get("seed", 42)))
    dist_env = init_distributed(str(train_cfg.distributed.get("backend", "none")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_cfg).to(device)
    load_checkpoint(resume_from, model=model)
    if dist_env.backend == "fsdp":
        model = wrap_fsdp(model, cfg=train_cfg.distributed.get("fsdp", {}))

    tokenizer = BPETokenizer.from_file(tokenizer_path)
    mix = [ToolMixEntry(name=str(e["name"]), weight=float(e["weight"])) for e in train_cfg.data.mix]
    dataset = ToolUseDataset(
        mix=mix,
        tokenizer=tokenizer,
        block_size=int(train_cfg.sequence_length),
        pad_token_id=tokenizer.pad_id(),
        mask_tool_results=bool(train_cfg.data.get("mask_tool_results", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.micro_batch_size),
        collate_fn=collate_sft_batch,
        num_workers=int(train_cfg.data.get("num_workers", 0)),
    )

    optimizer = build_optimizer(model.parameters(), train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, total_steps=int(train_cfg.max_steps))

    output_dir = Path(train_cfg.get("output_dir", "outputs/tool_use_sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(model_cfg, output_dir / "model_config.yaml")
    save_config(train_cfg, output_dir / "train_config.yaml")

    rolling = RollingMean()
    model.train()
    step = 0
    for batch in loader:
        if step >= int(train_cfg.max_steps):
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        rolling.update(float(loss.item()))
        step += 1
        if step % int(train_cfg.get("log_every", 25)) == 0 and dist_env.is_main:
            logger.info("tool-use sft step %d | loss %.4f", step, rolling.mean)
        if step % int(train_cfg.get("save_every", 1000)) == 0 and dist_env.is_main:
            save_checkpoint(output_dir=output_dir, step=step, model=model, optimizer=optimizer,
                            scheduler=scheduler, config=model_cfg)
    if dist_env.is_main:
        save_checkpoint(output_dir=output_dir, step=step, model=model, optimizer=optimizer,
                        scheduler=scheduler, config=model_cfg)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool-use SFT for Forge.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", default="tool-use-sft")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("override", nargs="*")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_cfg = load_config(f"models/{args.model}")
    train_cfg = load_config(f"training/{args.training}")
    train_cfg = override_from_cli(train_cfg, args.override)
    run_tool_use_sft(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer_path=args.tokenizer,
        resume_from=args.resume_from,
    )
