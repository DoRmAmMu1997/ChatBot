"""Smoke test for the RLEF reward + loss path on synthetic problems."""

from __future__ import annotations

import torch

from chatbot.data.synthetic_swebench import iter_synthetic_issues
from chatbot.training.rlef import RLEFRollout, _grpo_loss, _score_dry_run


def test_synthetic_issues_yield_with_required_fields():
    issues = list(iter_synthetic_issues(num=4))
    assert len(issues) == 4
    for issue in issues:
        assert issue.problem_statement
        assert issue.failing_test
        assert issue.buggy_function
        assert issue.fixed_function


def test_score_dry_run_partial_credit():
    issue = next(iter(iter_synthetic_issues(num=1)))
    # Containing the exact fix gives reward 1.0.
    assert _score_dry_run(issue, issue.fixed_function) == 1.0
    # An empty response gets 0.
    assert _score_dry_run(issue, "") == 0.0


def test_grpo_loss_shape():
    issue = next(iter(iter_synthetic_issues(num=1)))
    rollouts = [
        RLEFRollout(
            prompt_ids=torch.zeros(4, dtype=torch.long),
            response_ids=torch.zeros(2, dtype=torch.long),
            log_probs=torch.tensor([-1.0, -0.5], requires_grad=True),
            reward=float(reward),
        )
        for reward in (1.0, 0.0, 0.5, 0.0)
    ]
    loss = _grpo_loss(rollouts)
    assert loss.dim() == 0
