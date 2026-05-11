"""Execute all hooks registered for a given event in order."""

from __future__ import annotations

from typing import Dict

from .registry import HookRegistry


def run_hooks(registry: HookRegistry, event: str, payload: Dict) -> Dict:
    """Pass ``payload`` through every registered hook and return it.

    Hooks may either mutate ``payload`` in place or return a new dict that
    replaces it. Exceptions inside a hook propagate up — hooks are supposed
    to be cheap and trustworthy.
    """

    for hook in registry.hooks_for(event):
        result = hook(payload)
        if isinstance(result, dict):
            payload = result
    return payload
