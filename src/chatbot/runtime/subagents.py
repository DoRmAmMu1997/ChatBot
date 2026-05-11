"""Spawn isolated sub-agents with their own tool subsets.

This is the same pattern as Claude Code's ``Agent`` tool: when the parent
agent wants to dispatch a self-contained chunk of work (research, batch
edits, evaluation), it spawns a child agent that:

* sees only the system prompt + the launch message,
* has access to a configurable subset of tools,
* returns a single text result back to the parent.

The implementation here is intentionally lightweight — the child agent
shares the same model instance as the parent (we don't load the model
twice) but maintains its own message history and tool registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class SubagentSpec:
    """Description of a child agent to spawn."""

    description: str
    prompt: str
    allowed_tools: Optional[List[str]] = None
    max_turns: int = 8


class SubagentManager:
    """Tracks active sub-agents so the parent can cap concurrency."""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self._active: List["object"] = []

    def can_spawn(self) -> bool:
        return len(self._active) < self.max_concurrent

    def attach(self, agent: "object") -> None:
        self._active.append(agent)

    def detach(self, agent: "object") -> None:
        try:
            self._active.remove(agent)
        except ValueError:
            pass

    def active_count(self) -> int:
        return len(self._active)
