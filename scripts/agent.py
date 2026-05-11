"""Interactive agent REPL with plugins, skills, hooks, and slash commands."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chatbot.runtime.agent import cli  # noqa: E402


if __name__ == "__main__":
    cli()
