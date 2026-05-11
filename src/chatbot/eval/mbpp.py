"""MBPP runner — same shape as HumanEval but with a different prompt format."""

from __future__ import annotations

from typing import Any, Dict, List

from ..inference.batch_generate import batch_generate
from .humaneval import _safe_run
from .runner import EvalResult, register_benchmark


@register_benchmark("mbpp")
def run_mbpp(
    *,
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> EvalResult:
    """Score a model on MBPP pass@1.

    Each problem is ``{"text": ..., "test_list": [...]}``. We wrap the test
    list into a ``check()`` function so we can reuse the HumanEval runner.
    """

    if not problems:
        return EvalResult(benchmark="mbpp", score=0.0, num_examples=0,
                          notes=["No problems supplied — load the mbpp dataset."])

    prompts = [f"# {p['text']}\n" for p in problems]
    completions = batch_generate(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    passes = 0
    for problem, completion in zip(problems, completions):
        assertions = "\n".join(problem["test_list"])
        test = f"def check(_):\n" + "\n".join(f"    {a}" for a in problem["test_list"])
        # MBPP doesn't carry an entry-point name, so we exec the candidate alone
        # and run assertions afterwards. We synthesize a no-op `check`.
        try:
            if _safe_run(completion + "\n" + assertions, "def check(_): pass", "check"):
                passes += 1
        except Exception:
            continue
    score = passes / len(problems)
    return EvalResult(
        benchmark="mbpp", score=score, num_examples=len(problems),
        details={"passed": passes, "total": len(problems)},
    )
