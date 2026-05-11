"""Hooks for the example plugin."""

from __future__ import annotations

import logging

_LOG = logging.getLogger("plugin.filesystem_mcp_example")


def log_call(payload: dict) -> dict:
    """Pre-tool hook: log the call so we can trace agent behaviour."""

    _LOG.info("Tool %s called with %s", payload.get("tool_name"), payload.get("arguments"))
    return payload
