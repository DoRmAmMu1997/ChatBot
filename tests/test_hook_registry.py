"""Hook registry + runner tests."""

from __future__ import annotations

import pytest

from chatbot.runtime.hooks.registry import HookRegistry
from chatbot.runtime.hooks.runner import run_hooks


def test_hooks_run_in_registration_order():
    registry = HookRegistry()
    log = []

    registry.register("pre_tool", lambda p: log.append("a") or p)
    registry.register("pre_tool", lambda p: log.append("b") or p)
    registry.register("pre_tool", lambda p: log.append("c") or p)

    run_hooks(registry, "pre_tool", {})
    assert log == ["a", "b", "c"]


def test_hook_can_replace_payload():
    registry = HookRegistry()
    registry.register("pre_tool", lambda p: {"replaced": True})
    out = run_hooks(registry, "pre_tool", {"orig": True})
    assert out == {"replaced": True}


def test_unknown_event_raises():
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.register("bogus", lambda p: p)
