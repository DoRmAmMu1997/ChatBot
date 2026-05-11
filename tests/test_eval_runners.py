"""Smoke tests for the benchmark runners (with empty datasets)."""

from __future__ import annotations

import torch
import torch.nn as nn

from chatbot.eval import run_gsm8k, run_humaneval, run_mbpp, run_mmlu


class _Identity(nn.Module):
    """Dummy LM that just returns zero logits — fine for smoke testing."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x, **kwargs):
        batch, seq = x.shape
        return {"logits": torch.zeros(batch, seq, self.vocab_size)}


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * max(1, len(text) // 4)

    def decode(self, ids, skip_special_tokens=True):
        return ""

    def eos_id(self):
        return 0


def test_humaneval_with_no_problems():
    res = run_humaneval(model=_Identity(8), tokenizer=_FakeTokenizer(), problems=[])
    assert res.score == 0.0
    assert res.num_examples == 0


def test_mbpp_with_no_problems():
    res = run_mbpp(model=_Identity(8), tokenizer=_FakeTokenizer(), problems=[])
    assert res.score == 0.0
    assert res.num_examples == 0


def test_mmlu_with_no_problems():
    res = run_mmlu(model=_Identity(8), tokenizer=_FakeTokenizer(), questions=[])
    assert res.score == 0.0


def test_gsm8k_with_no_problems():
    res = run_gsm8k(model=_Identity(8), tokenizer=_FakeTokenizer(), problems=[])
    assert res.score == 0.0
