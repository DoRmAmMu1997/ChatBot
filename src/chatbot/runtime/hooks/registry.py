"""Lifecycle hook registry.

A hook is a plain callable that is invoked at a named lifecycle event. Each
hook may *mutate* the event payload (a dict) in place to influence what
the agent does next. The supported events are:

* ``pre_tool``: called with ``{"tool": Tool, "arguments": dict}`` right
  before a tool is dispatched. Setting ``payload["skip"] = True`` cancels
  the call; setting ``payload["arguments"] = ...`` rewrites them.
* ``post_tool``: called with ``{"tool": Tool, "arguments": dict, "result": Any}``
  right after a tool returns. Hook can rewrite ``result``.
* ``on_message``: called with ``{"message": AgentMessage}`` whenever the
  agent observes a new message.
* ``on_stop``: called with ``{"history": list, "reason": str}`` when the
  agent finishes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


HookFn = Callable[[Dict], Optional[Dict]]
SUPPORTED_EVENTS = {"pre_tool", "post_tool", "on_message", "on_stop"}


@dataclass
class _Entry:
    fn: HookFn
    plugin: Optional[str] = None


class HookRegistry:
    def __init__(self):
        self._by_event: Dict[str, List[_Entry]] = {event: [] for event in SUPPORTED_EVENTS}

    def register(self, event: str, fn: HookFn, *, plugin: Optional[str] = None) -> None:
        if event not in SUPPORTED_EVENTS:
            raise ValueError(f"Unknown hook event: {event!r}. Allowed: {sorted(SUPPORTED_EVENTS)}")
        self._by_event[event].append(_Entry(fn=fn, plugin=plugin))

    def hooks_for(self, event: str) -> List[HookFn]:
        return [e.fn for e in self._by_event.get(event, [])]
