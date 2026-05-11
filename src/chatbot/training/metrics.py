"""Tiny training metric helpers — rolling means and simple counters."""

from __future__ import annotations

from collections import deque
from typing import Deque


class RollingMean:
    """Average of the last ``window`` values."""

    def __init__(self, window: int = 50):
        self.window = window
        self._values: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._values.append(float(value))

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)
