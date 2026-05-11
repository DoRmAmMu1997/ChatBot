"""CLI wrapper around chatbot.tokenizer.train_tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chatbot.tokenizer.train_tokenizer import main  # noqa: E402


if __name__ == "__main__":
    main()
