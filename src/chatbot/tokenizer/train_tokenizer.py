"""Programmatic entry point for training a BPE tokenizer on a corpus.

CLI thin wrapper lives at ``scripts/train_tokenizer.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .bpe import BPETokenizer, SpecialTokens


def train_tokenizer(
    files: Sequence[str],
    *,
    vocab_size: int,
    output_path: str,
    min_frequency: int = 2,
) -> BPETokenizer:
    tokenizer = BPETokenizer.train(
        files=files,
        vocab_size=vocab_size,
        specials=SpecialTokens(),
        min_frequency=min_frequency,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    return tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--files", nargs="+", required=True, help="Text files to train on.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output", required=True, help="Where to save tokenizer.json.")
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_tokenizer(
        files=args.files,
        vocab_size=args.vocab_size,
        output_path=args.output,
        min_frequency=args.min_frequency,
    )
