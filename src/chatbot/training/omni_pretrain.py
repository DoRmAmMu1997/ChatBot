"""Omni-modal next-token-prediction training.

The same loss as text-only pretrain, but the data loader streams sequences
that include interleaved image and audio placeholders. The model splices
in modality embeddings before the forward pass — labels mask placeholder
positions so the LM doesn't try to predict the visual / audio tokens
themselves.

For now we delegate the heavy lifting to ``training.pretrain.run_pretrain``
and just provide a thin wrapper that ensures the trainer model class
supports the new modality kwargs. (The Aurora/Forge classes both do.)
"""

from __future__ import annotations

import argparse

from ..utils.config import load_config, override_from_cli
from .pretrain import run_pretrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Omni-modal pretrain (text + image + audio).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--training", default="omni_pretrain")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    model_cfg = load_config(f"models/{args.model}")
    train_cfg = load_config(f"training/{args.training}")
    train_cfg = override_from_cli(train_cfg, args.override)
    run_pretrain(model_cfg=model_cfg, train_cfg=train_cfg, tokenizer_path=args.tokenizer)
