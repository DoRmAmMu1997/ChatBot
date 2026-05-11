"""Slash-command handlers for the example plugin."""

from __future__ import annotations

import os


def current_dir(_rest: str) -> str:
    """Return a one-line description of the current working directory."""

    return f"Current working directory: {os.getcwd()}"
