"""GSM8K — grade-school math word problems. Match the final numeric answer."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..inference.batch_generate import batch_generate
from .runner import EvalResult, register_benchmark


_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,\.]*)")
_FALLBACK_RE = re.compile(r"(-?\d[\d,\.]*)")


def _extract_number(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number anywhere in the text.
    nums = _FALLBACK_RE.findall(text)
    return nums[-1].replace(",", "") if nums else ""


@register_benchmark("gsm8k")
def run_gsm8k(
    *,
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> EvalResult:
    if not problems:
        return EvalResult(benchmark="gsm8k", score=0.0, num_examples=0,
                          notes=["No problems supplied — load openai/gsm8k."])
    prompts = [
        f"Solve the problem step by step. End with '#### <answer>'.\n\n"
        f"Problem: {p['question']}\nSolution:"
        for p in problems
    ]
    completions = batch_generate(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
    )
    correct = 0
    for completion, problem in zip(completions, problems):
        gold = _extract_number(problem["answer"])
        pred = _extract_number(completion)
        if pred and gold and pred == gold:
            correct += 1
    return EvalResult(
        benchmark="gsm8k",
        score=correct / len(problems),
        num_examples=len(problems),
        details={"correct": correct},
    )
