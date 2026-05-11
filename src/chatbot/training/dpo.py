"""Direct Preference Optimization (DPO) training loop.

DPO's loss is::

    L = -log sigmoid( beta * (log pi(chosen|x) - log pi_ref(chosen|x)
                              - log pi(rejected|x) + log pi_ref(rejected|x)) )

In plain English: we're trying to make the policy model assign HIGHER
log-probability to the chosen response than the rejected one, relative to a
frozen *reference* model (typically the SFT checkpoint). ``beta`` controls
how strongly we push — smaller is gentler.

Why no separate reward model: the algebra shows that an optimal reward
model + RL becomes equivalent to this direct preference objective. Less
machinery, more stable training.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..data.dpo_loader import DPODataset, DPOMixEntry, collate_dpo_batch
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


def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Sum the model's log-probabilities of the label tokens (ignoring padding).

    Returns ``log p(label_sequence | prompt)`` per batch row — the
    quantity DPO compares between policy and reference.

    Args:
        logits: ``[batch, seq, vocab]`` — the model's per-position output.
        labels: ``[batch, seq]`` — the token ids whose log-prob we want.
        pad_id: positions with this id are masked out so padding doesn't
            contribute to the sum.
    """

    # ``log_softmax`` is numerically more stable than ``log(softmax(...))``;
    # we cast to float32 because DPO's losses are sensitive to precision.
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # ``labels.clamp_min(0)`` keeps the gather safe even on padding rows
    # (where the label might be -100 or some other sentinel); we'll
    # zero out those contributions with the mask below.
    chosen_lp = log_probs.gather(dim=-1, index=labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    # Mask out pad positions — they shouldn't influence the sequence
    # log-likelihood.
    mask = (labels != pad_id).float()
    return (chosen_lp * mask).sum(dim=-1)


def run_dpo(
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

    policy = build_model(model_cfg).to(device)
    reference = build_model(model_cfg).to(device)
    load_checkpoint(resume_from, model=policy)
    load_checkpoint(resume_from, model=reference)
    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()

    if dist_env.backend == "fsdp":
        policy = wrap_fsdp(policy, cfg=train_cfg.distributed.get("fsdp", {}))

    tokenizer = BPETokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.pad_id()
    mix = [DPOMixEntry(name=str(e["name"]), weight=float(e["weight"])) for e in train_cfg.data.mix]
    dataset = DPODataset(
        mix=mix,
        tokenizer=tokenizer,
        max_prompt_length=int(train_cfg.sequence_length) // 2,
        max_response_length=int(train_cfg.sequence_length) // 2,
        pad_token_id=pad_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.micro_batch_size),
        collate_fn=lambda b: collate_dpo_batch(b, pad_id=pad_id),
        num_workers=int(train_cfg.data.get("num_workers", 0)),
    )

    optimizer = build_optimizer(policy.parameters(), train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler, total_steps=int(train_cfg.max_steps))

    output_dir = Path(train_cfg.get("output_dir", "outputs/dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(model_cfg, output_dir / "model_config.yaml")
    save_config(train_cfg, output_dir / "train_config.yaml")

    beta = float(train_cfg.dpo.get("beta", 0.1))
    rolling = RollingMean()

    policy.train()
    step = 0
    for batch in loader:
        if step >= int(train_cfg.max_steps):
            break
        prompt = batch["prompt_ids"].to(device)
        chosen = batch["chosen_ids"].to(device)
        rejected = batch["rejected_ids"].to(device)

        # Build the full sequences (prompt + response) for chosen and
        # rejected. We score the *response* portion only when computing
        # the log-probability, but the model needs the prompt context.
        chosen_full = torch.cat([prompt, chosen], dim=1)
        rejected_full = torch.cat([prompt, rejected], dim=1)

        # Policy log-probabilities — gradient flows through these.
        pol_chosen = _gather_logprobs(policy(chosen_full)["logits"], chosen_full, pad_id)
        pol_rejected = _gather_logprobs(policy(rejected_full)["logits"], rejected_full, pad_id)

        # Reference log-probabilities — frozen, so no grad. The reference
        # is typically the SFT checkpoint we started from.
        with torch.no_grad():
            ref_chosen = _gather_logprobs(reference(chosen_full)["logits"], chosen_full, pad_id)
            ref_rejected = _gather_logprobs(reference(rejected_full)["logits"], rejected_full, pad_id)

        # DPO's core scoring quantity. Intuition: how much *more* the
        # policy prefers ``chosen`` over ``rejected`` than the reference
        # does. Positive ``logits`` means we're already aligned with the
        # preference; negative means we need to swing more toward
        # ``chosen``. ``beta`` controls how strongly we push — smaller
        # beta keeps the policy closer to the reference.
        logits = beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
        loss = -F.logsigmoid(logits).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(train_cfg.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        rolling.update(float(loss.item()))
        step += 1
        if step % int(train_cfg.get("log_every", 25)) == 0 and dist_env.is_main:
            logger.info("dpo step %d | loss %.4f", step, rolling.mean)
        if step % int(train_cfg.get("save_every", 1000)) == 0 and dist_env.is_main:
            save_checkpoint(output_dir=output_dir, step=step, model=policy, optimizer=optimizer,
                            scheduler=scheduler, config=model_cfg)
    if dist_env.is_main:
        save_checkpoint(output_dir=output_dir, step=step, model=policy, optimizer=optimizer,
                        scheduler=scheduler, config=model_cfg)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO fine-tune from an SFT checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", default="dpo")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("override", nargs="*")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_cfg = load_config(f"models/{args.model}")
    train_cfg = load_config(f"training/{args.training}")
    train_cfg = override_from_cli(train_cfg, args.override)
    run_dpo(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer_path=args.tokenizer,
        resume_from=args.resume_from,
    )
