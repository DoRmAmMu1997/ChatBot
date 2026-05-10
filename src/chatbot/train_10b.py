"""ChatBot-10B training wrapper."""

from __future__ import annotations

import argparse

from .train import build_arg_parser, train


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Train the original untrained ChatBot-10B model."
    parser.set_defaults(
        config="configs/chatbot-10b.yaml",
        tokenizer="bpe",
        tokenizer_path="tokenizers/chatbot-bpe.json",
        dataset="mixed",
        checkpoint_name="chatbot-10b-untrained-init.pt",
    )
    args = parser.parse_args()
    if args.cpu:
        print("Warning: ChatBot-10B is not practical on CPU; use configs/chatbot-tiny.yaml for smoke tests.")
    train(args)


if __name__ == "__main__":
    main()
