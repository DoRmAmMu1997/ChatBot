"""Locally-generated agent traces against canned SWE-bench-style problems.

Why this exists: SWE-Bench-Verified's training split is tiny (~150 issues).
For RLEF training we want many more rollouts to draw rewards from. This
module produces a stream of *templated* (problem, solution) pairs designed
to exercise the same skills:

* read a bug report,
* look at the failing test,
* edit a single function,
* re-run the test.

The "ground-truth" patches are synthesized algorithmically, not by humans,
so they're not a substitute for real SWE-Bench data. They are a useful
supplementary signal during agentic RL — the rollouts that produce passing
patches get reward 1.
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass
from typing import Iterator, List


@dataclass(frozen=True)
class SyntheticIssue:
    """One synthetic SWE-bench-style problem."""

    issue_id: str
    repo: str
    problem_statement: str
    failing_test: str         # pytest snippet that fails on the buggy code.
    buggy_function: str       # buggy version of the function under test.
    fixed_function: str       # correct version (used to verify rollouts).
    failing_filename: str = "buggy.py"
    test_filename: str = "test_buggy.py"


_TEMPLATES = [
    {
        "topic": "off-by-one",
        "problem": "The function returns one less element than expected when slicing a list.",
        "buggy": "def take_first_n(items, n):\n    return items[:n - 1]\n",
        "fixed": "def take_first_n(items, n):\n    return items[:n]\n",
        "tests": "from buggy import take_first_n\n\n"
                  "def test_take_first_n():\n"
                  "    assert take_first_n([1, 2, 3], 2) == [1, 2]\n",
    },
    {
        "topic": "string formatting",
        "problem": "Printing user names produces an extra trailing space.",
        "buggy": "def format_name(first, last):\n    return f'{first} {last} '\n",
        "fixed": "def format_name(first, last):\n    return f'{first} {last}'\n",
        "tests": "from buggy import format_name\n\n"
                  "def test_format_name():\n"
                  "    assert format_name('ada', 'lovelace') == 'ada lovelace'\n",
    },
    {
        "topic": "division by zero",
        "problem": "Computing rate crashes when total is zero.",
        "buggy": "def rate(part, total):\n    return part / total\n",
        "fixed": "def rate(part, total):\n    if total == 0:\n        return 0.0\n    return part / total\n",
        "tests": "from buggy import rate\n\n"
                  "def test_rate_zero_total():\n"
                  "    assert rate(5, 0) == 0.0\n"
                  "def test_rate_normal():\n"
                  "    assert rate(1, 4) == 0.25\n",
    },
    {
        "topic": "wrong return type",
        "problem": "Function returns the wrong container type; callers expect a list.",
        "buggy": "def deduplicate(items):\n    return set(items)\n",
        "fixed": "def deduplicate(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        "tests": "from buggy import deduplicate\n\n"
                  "def test_deduplicate():\n"
                  "    assert deduplicate([1, 2, 2, 3]) == [1, 2, 3]\n",
    },
]


def iter_synthetic_issues(*, seed: int = 0, num: int = 1000) -> Iterator[SyntheticIssue]:
    """Yield up to ``num`` synthetic issues sampled from the templates."""

    rng = random.Random(seed)
    for i in range(num):
        template = rng.choice(_TEMPLATES)
        repo = rng.choice([
            "acme/inventory", "acme/payments", "openai/widgets",
            "fortylabs/billing", "djangonauts/blog", "scifi/orchestra",
        ])
        problem = textwrap.dedent(f"""\
            ## Bug report — {template['topic']}

            {template['problem']}

            Reproduction: see the failing test in ``test_buggy.py``.
            Fix the implementation in ``buggy.py`` so the test passes.
            """)
        yield SyntheticIssue(
            issue_id=f"synthetic-{i:05d}",
            repo=repo,
            problem_statement=problem,
            failing_test=template["tests"],
            buggy_function=template["buggy"],
            fixed_function=template["fixed"],
        )


def issue_to_chat_messages(issue: SyntheticIssue) -> List[dict]:
    """Render an issue as a chat-message list that an agent would receive."""

    return [
        {"role": "system", "content": "You are a senior software engineer fixing a bug."},
        {"role": "user", "content": (
            issue.problem_statement
            + "\n\n=== file: buggy.py ===\n"
            + issue.buggy_function
            + "\n=== file: test_buggy.py ===\n"
            + issue.failing_test
        )},
    ]
