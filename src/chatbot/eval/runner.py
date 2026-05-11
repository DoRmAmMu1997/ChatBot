"""Common dispatcher used by ``scripts/eval.py``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class EvalResult:
    benchmark: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    num_examples: int = 0
    notes: List[str] = field(default_factory=list)


BENCHMARK_RUNNERS: Dict[str, Callable[..., EvalResult]] = {}


def register_benchmark(name: str):
    def decorator(fn: Callable[..., EvalResult]):
        BENCHMARK_RUNNERS[name] = fn
        return fn
    return decorator


def run_benchmark(name: str, **kwargs) -> EvalResult:
    if name not in BENCHMARK_RUNNERS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(BENCHMARK_RUNNERS)}")
    return BENCHMARK_RUNNERS[name](**kwargs)
