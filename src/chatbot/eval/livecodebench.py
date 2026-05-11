"""LiveCodeBench runner — contemporary competitive-programming problems.

The official LiveCodeBench dataset is updated over time. We re-use the
HumanEval-style sandboxed runner because each problem ships with a
``check`` function or assertion list, just like HumanEval.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..inference.batch_generate import batch_generate
from .humaneval import _safe_run
from .runner import EvalResult, register_benchmark


@register_benchmark("livecodebench")
def run_livecodebench(
    *,
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> EvalResult:
    if not problems:
        return EvalResult(benchmark="livecodebench", score=0.0, num_examples=0,
                          notes=["No problems supplied — load livecodebench/code_generation."])
    prompts = [p["prompt"] for p in problems]
    completions = batch_generate(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature,
    )
    passes = 0
    for problem, completion in zip(problems, completions):
        full_code = problem["prompt"] + completion
        if _safe_run(full_code, problem["test"], problem.get("entry_point", "solution")):
            passes += 1
    return EvalResult(
        benchmark="livecodebench",
        score=passes / len(problems),
        num_examples=len(problems),
        details={"passed": passes},
    )
