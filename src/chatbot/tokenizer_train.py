"""Command-line helpers for training the BPE tokenizer."""

from __future__ import annotations

import argparse
import os

from .data import load_training_texts
from .tokenizer import BPETokenizer, write_tokenizer_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for ChatBot.")

    # Dataset flags match train.py so the tokenizer can be trained on the same
    # text distribution the model will later learn from.
    parser.add_argument("--dataset", default="mixed")
    parser.add_argument("--corpus-dir", default=os.path.join("data", "cornell movie-dialogs corpus"))
    parser.add_argument("--hf-dataset", default="OpenRL/daily_dialog")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--vocab-size", type=int, default=128000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", default=os.path.join("tokenizers", "chatbot-bpe.json"))
    parser.add_argument("--manifest", default=os.path.join("tokenizers", "chatbot-bpe-manifest.json"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # First collect formatted conversation strings such as:
    # <bos> <user> hello <bot> hi <eos>
    texts = load_training_texts(
        dataset=args.dataset,
        corpus_dir=args.corpus_dir,
        max_pairs=args.max_pairs,
        hf_dataset_name=args.hf_dataset,
        hf_split=args.hf_split,
    )

    # Then learn subword pieces from those strings and save the tokenizer for
    # training/inference. The model and tokenizer must stay paired.
    tokenizer = BPETokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.output)
    write_tokenizer_manifest(args.manifest, tokenizer_path=args.output, vocab_size=tokenizer.vocab_size)
    print(f"Saved BPE tokenizer to {args.output}")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")


if __name__ == "__main__":
    main()
