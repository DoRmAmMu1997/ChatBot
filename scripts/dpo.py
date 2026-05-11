"""CLI wrapper around chatbot.training.dpo."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chatbot.training.dpo import main  # noqa: E402


if __name__ == "__main__":
    main()
