"""Plain Python logging set up the way we like it for training runs."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: Optional[str] = None,
    rank: Optional[int] = None,
) -> None:
    """Configure the root logger for training and inference.

    In multi-rank training, the env var ``RANK`` is set by ``torchrun``. We
    silence non-zero ranks by default so the console isn't spammed with
    duplicate lines — they still log to file if one is provided.
    """

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    actual_rank = rank if rank is not None else int(os.environ.get("RANK", "0"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    if actual_rank == 0:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        stream.setLevel(level)
        root.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        file_handler.setLevel(level)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger using the project's settings."""

    return logging.getLogger(name)
