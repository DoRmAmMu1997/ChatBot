"""Custom tool exposed by the example plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


word_count__description = "Count words in a UTF-8 text file."


def word_count(args: Dict[str, Any]) -> Dict[str, Any]:
    text = Path(args["path"]).read_text(encoding="utf-8")
    # Splitting on whitespace is a fine word count for the common case;
    # for production NLP you'd want a proper tokenizer.
    return {"words": len(text.split()), "chars": len(text)}
