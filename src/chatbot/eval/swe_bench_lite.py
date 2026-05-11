"""SWE-bench Lite runner — agent-mode end-to-end software engineering.

This is the hardest of the benchmarks we ship. SWE-bench gives the model a
real GitHub issue + the corresponding repo at a fixed commit, asks for a
patch, and grades by running the project's test suite before and after.

Running this benchmark properly needs containerized environments per
problem (the original SWE-bench provides Docker images). Our runner here
focuses on the patch-generation half — we drop the candidate patch into
the working tree but expect the user to wire up the actual test-running
step in their environment, since that requires a sandboxed Docker host.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..runtime.agent import Agent
from .runner import EvalResult, register_benchmark


@register_benchmark("swe_bench_lite")
def run_swe_bench_lite(
    *,
    agent: Agent,
    problems: List[Dict[str, Any]],
    apply_patch_fn=None,
    run_tests_fn=None,
) -> EvalResult:
    """Score the agent on SWE-bench Lite.

    Args:
        agent: a fully-built :class:`Agent` (Forge with filesystem + shell tools).
        problems: list of SWE-bench instances (``problem_statement``, ``repo``,
            ``base_commit``, ``test_patch``, ``FAIL_TO_PASS``, ``PASS_TO_PASS``).
        apply_patch_fn: callback ``(problem, patch_text) -> bool``; user-supplied,
            because checking out a repo at a specific commit requires git
            tooling we don't ship.
        run_tests_fn: callback ``(problem) -> bool``; returns True if all the
            FAIL_TO_PASS + PASS_TO_PASS tests pass after the patch.
    """

    if not problems:
        return EvalResult(benchmark="swe_bench_lite", score=0.0, num_examples=0,
                          notes=["No problems supplied — load princeton-nlp/SWE-bench_Lite."])
    if apply_patch_fn is None or run_tests_fn is None:
        return EvalResult(
            benchmark="swe_bench_lite", score=0.0, num_examples=len(problems),
            notes=["No apply_patch_fn / run_tests_fn callbacks supplied — running in dry mode."],
        )

    passes = 0
    for problem in problems:
        prompt = (
            "You are working in the repository "
            f"{problem.get('repo', '<repo>')}. The current issue:\n\n"
            f"{problem.get('problem_statement', '')}\n\n"
            "Use your tools to read the codebase and produce a minimal patch "
            "that fixes the issue. When done, print the unified diff between "
            "<patch> and </patch> tags."
        )
        reply = agent.respond(prompt)
        patch = _extract_patch(reply)
        if not patch:
            continue
        if apply_patch_fn(problem, patch) and run_tests_fn(problem):
            passes += 1

    return EvalResult(
        benchmark="swe_bench_lite",
        score=passes / len(problems),
        num_examples=len(problems),
        details={"passed": passes},
    )


def _extract_patch(text: str) -> str:
    """Pull the content between ``<patch>...</patch>`` tags out of a reply."""

    start = text.find("<patch>")
    end = text.find("</patch>", start + 1)
    if start == -1 or end == -1:
        return ""
    return text[start + len("<patch>") : end].strip()
