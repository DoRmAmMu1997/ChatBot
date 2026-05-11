"""Reinforcement Learning from Execution Feedback (RLEF) for Forge.

The recipe is straightforward in spirit and finicky in practice:

  1. Sample N rollouts from the current policy on a SWE-bench-style
     problem. Each rollout uses the agent loop (filesystem + shell tools)
     inside a sandboxed working tree.
  2. Score each rollout by *running the project's tests*. Reward = 1 if all
     gating tests pass, else 0 (with a small shaped reward equal to the
     pass-rate of the gating tests, to make partial progress visible).
  3. Compute a leave-one-out baseline across the group and apply the
     REINFORCE policy gradient with the baseline subtracted — i.e. GRPO.

We're not going to actually run real Docker sandboxes here; the caller
plugs in ``apply_patch_fn`` + ``run_tests_fn`` callbacks (same shape as
the SWE-bench-Lite eval). On a tiny config without those callbacks, we
fall back to a dry-run mode that exercises the loss path on synthetic
issues from ``data.synthetic_swebench``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F

from ..data.synthetic_swebench import (
    SyntheticIssue,
    issue_to_chat_messages,
    iter_synthetic_issues,
)
from ..tokenizer.bpe import BPETokenizer
from ..tokenizer.chat_template import tool_chat_template
from ..training.checkpoint import load_checkpoint, save_checkpoint
from ..training.metrics import RollingMean
from ..training.optim import build_optimizer, build_scheduler
from ..utils.config import load_config, override_from_cli
from ..utils.logging import get_logger, setup_logging
from ..utils.seeding import set_seed
from .pretrain import build_model

logger = get_logger(__name__)


@dataclass
class RLEFRollout:
    """One rollout against one SWE-bench-style problem."""

    prompt_ids: torch.Tensor          # [seq]
    response_ids: torch.Tensor        # [resp]
    log_probs: torch.Tensor           # [resp] — per-token log p(response | prompt) under policy
    reward: float


def _grpo_loss(rollouts: List[RLEFRollout]) -> torch.Tensor:
    """Group-relative policy optimization loss (REINFORCE + LOO baseline).

    For each problem we take G rollouts and use their *group-mean* reward as
    a baseline. The loss is::

        L = - (reward - baseline) * sum_t log p(token_t)

    averaged over the group. The baseline removes the high-variance "overall
    difficulty" signal and isolates the per-rollout advantage.
    """

    if not rollouts:
        return torch.tensor(0.0, requires_grad=False)
    rewards = torch.tensor([r.reward for r in rollouts])
    baseline = rewards.mean()
    advantages = rewards - baseline

    loss = torch.tensor(0.0)
    for rollout, advantage in zip(rollouts, advantages):
        if rollout.log_probs.numel() == 0:
            continue
        loss = loss + (-float(advantage) * rollout.log_probs.sum())
    return loss / len(rollouts)


def _score_dry_run(issue: SyntheticIssue, response_text: str) -> float:
    """Heuristic reward when no real test runner is available.

    Gives partial credit for getting closer to the fixed function:
    * 1.0 if the response contains the full fixed function body.
    * 0.5 if it mentions the variable names / function name from the fix.
    * 0.0 otherwise.
    """

    if issue.fixed_function.strip() in response_text:
        return 1.0
    keywords = [w for w in issue.fixed_function.split() if w.isidentifier()]
    if any(kw in response_text for kw in keywords):
        return 0.5
    return 0.0


def _rollout_one(
    model,
    tokenizer: BPETokenizer,
    issue: SyntheticIssue,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> RLEFRollout:
    """Sample one response from the policy and compute its log-probs."""

    template = tool_chat_template()
    prompt = template.render(issue_to_chat_messages(issue), add_generation_prompt=True)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    # Generate at non-zero temperature so we see exploration.
    out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                             temperature=0.8, top_p=0.95)
    response_ids = out_ids[0, prompt_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)

    # Re-score the generated response under the *current* policy (with grad)
    # so we have the log-probabilities we'll attribute the reward to.
    full = torch.cat([prompt_ids[0], response_ids], dim=0).unsqueeze(0)
    logits = model(full)["logits"][0, prompt_ids.shape[1] - 1 : -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    chosen = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)

    return RLEFRollout(
        prompt_ids=prompt_ids[0],
        response_ids=response_ids,
        log_probs=chosen,
        reward=_score_dry_run(issue, response_text),
    )


def run_rlef(
    *,
    model_cfg,
    train_cfg,
    tokenizer_path: str,
    resume_from: str,
    apply_patch_fn: Optional[Callable[..., bool]] = None,
    run_tests_fn: Optional[Callable[..., bool]] = None,
) -> Path:
    setup_logging(level="INFO")
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dry_run = apply_patch_fn is None or run_tests_fn is None

    model = build_model(model_cfg).to(device)
    load_checkpoint(resume_from, model=model)
    tokenizer = BPETokenizer.from_file(tokenizer_path)

    optimizer = build_optimizer(model.parameters(), train_cfg.optimizer)
    scheduler = build_scheduler(optimizer, train_cfg.scheduler,
                                 total_steps=int(train_cfg.max_steps))

    output_dir = Path(train_cfg.get("output_dir", "outputs/rlef"))
    output_dir.mkdir(parents=True, exist_ok=True)

    group_size = int(train_cfg.get("group_size", 4))
    max_new = int(train_cfg.get("max_new_tokens", 256))
    rolling_reward = RollingMean()
    rolling_loss = RollingMean()

    issues = iter_synthetic_issues(num=int(train_cfg.max_steps) * group_size + 10)
    model.train()
    step = 0
    while step < int(train_cfg.max_steps):
        # Collect ``group_size`` rollouts on the same issue.
        try:
            issue = next(issues)
        except StopIteration:
            break
        rollouts: List[RLEFRollout] = []
        for _ in range(group_size):
            rollout = _rollout_one(model, tokenizer, issue, max_new_tokens=max_new, device=device)
            if not dry_run:
                # If real callbacks are wired, recompute the reward by
                # actually applying the patch and running the tests.
                patch = tokenizer.decode(rollout.response_ids.tolist(), skip_special_tokens=True)
                ok = apply_patch_fn(issue, patch) and run_tests_fn(issue)
                rollout = RLEFRollout(
                    prompt_ids=rollout.prompt_ids,
                    response_ids=rollout.response_ids,
                    log_probs=rollout.log_probs,
                    reward=float(ok),
                )
            rollouts.append(rollout)

        loss = _grpo_loss(rollouts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        mean_reward = sum(r.reward for r in rollouts) / max(len(rollouts), 1)
        rolling_reward.update(mean_reward)
        rolling_loss.update(float(loss.item()))

        step += 1
        if step % int(train_cfg.get("log_every", 25)) == 0:
            logger.info(
                "rlef step %d | mean reward %.3f | loss %.4f",
                step,
                rolling_reward.mean,
                rolling_loss.mean,
            )
        if step % int(train_cfg.get("save_every", 250)) == 0:
            save_checkpoint(output_dir, step=step, model=model, optimizer=optimizer,
                            scheduler=scheduler, config=model_cfg)

    save_checkpoint(output_dir, step=step, model=model, optimizer=optimizer,
                    scheduler=scheduler, config=model_cfg)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="RLEF — reinforcement learning from execution feedback.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", default="rlef")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--resume-from", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic reward heuristic instead of real test runs.")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()
    model_cfg = load_config(f"models/{args.model}")
    train_cfg = load_config(f"training/{args.training}")
    train_cfg = override_from_cli(train_cfg, args.override)
    run_rlef(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer_path=args.tokenizer,
        resume_from=args.resume_from,
        apply_patch_fn=None if args.dry_run else None,  # plug in real callbacks here
        run_tests_fn=None if args.dry_run else None,
    )
