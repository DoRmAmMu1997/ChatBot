"""HumanEval runner.

HumanEval is OpenAI's hand-written set of 164 Python programming problems.
Each problem gives a function signature and docstring; the model has to
produce a body. We execute the model's solution against the problem's
unit tests and count pass@1.

Note: executing model-generated code is unsafe by default. We sandbox by
running each candidate in a subprocess with no network. For real
production evals, use a container or the official ``human-eval`` package
with its sandbox helpers.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import textwrap
from typing import Any, Dict, List, Optional

from ..inference.batch_generate import batch_generate
from .runner import EvalResult, register_benchmark


def _run_candidate(candidate: str, test: str, entry_point: str, queue) -> None:
    """Run candidate + test in a child process. Send ``True``/``False`` via queue."""

    namespace: Dict[str, Any] = {}
    try:
        exec(candidate, namespace)
        exec(test, namespace)
        check = namespace.get("check")
        if check is None:
            queue.put(False)
            return
        check(namespace[entry_point])
        queue.put(True)
    except BaseException:
        queue.put(False)


def _safe_run(candidate: str, test: str, entry_point: str, timeout: float = 10.0) -> bool:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_candidate, args=(candidate, test, entry_point, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
        return False
    try:
        return bool(queue.get_nowait())
    except Exception:
        return False


@register_benchmark("humaneval")
def run_humaneval(
    *,
    model,
    tokenizer,
    problems: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> EvalResult:
    """Score a model on HumanEval pass@1.

    ``problems`` is a list of dicts with keys ``prompt``, ``test``,
    ``entry_point`` — matching the layout in ``openai_humaneval``.
    """

    if not problems:
        return EvalResult(benchmark="humaneval", score=0.0, num_examples=0,
                          notes=["No problems supplied — load the openai_humaneval dataset."])

    prompts = [p["prompt"] for p in problems]
    completions = batch_generate(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, temperature=temperature,
    )
    passes = 0
    for problem, completion in zip(problems, completions):
        full_code = problem["prompt"] + completion
        if _safe_run(full_code, problem["test"], problem["entry_point"]):
            passes += 1
    score = passes / len(problems)
    return EvalResult(
        benchmark="humaneval",
        score=score,
        num_examples=len(problems),
        details={"passed": passes, "total": len(problems)},
    )
