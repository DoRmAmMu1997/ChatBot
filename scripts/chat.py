"""Interactive chat CLI. Thin wrapper around chatbot.inference.generate.cli."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chatbot.inference.generate import cli  # noqa: E402


if __name__ == "__main__":
    cli()
