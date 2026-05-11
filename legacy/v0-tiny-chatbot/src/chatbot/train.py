"""Training loop for the small Transformer chatbot."""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import build_datasets, load_training_texts
from .model import TransformerChatModel
from .tokenizer import SimpleTokenizer


def set_seed(seed: int) -> None:
    """Make training runs easier to reproduce.

    Randomness is used for weight initialization, data shuffling, and sampling.
    Setting the seed makes quick experiments easier to compare.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(
    model: TransformerChatModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """Measure validation loss on a small number of batches.

    Validation loss is computed on examples the model is not updating from. It
    gives a rough signal for whether the model is learning patterns that
    generalize beyond the current training batches.
    """

    model.eval()
    losses = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        # Passing y asks the model to return both logits and loss. During
        # validation we read the loss but do not call backward().
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(
    checkpoint_path: str,
    model: TransformerChatModel,
    tokenizer: SimpleTokenizer,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    """Save everything needed for later chat inference.

    The checkpoint stores more than weights. It also stores the tokenizer and
    config, because inference must use the exact same vocabulary and model size
    that training used.
    """

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "train_args": vars(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def train(args: argparse.Namespace) -> str:
    """Train the model and return the checkpoint path.

    The high-level flow is:
    1. load text pairs,
    2. build a tokenizer,
    3. turn text into token batches,
    4. train next-token prediction,
    5. save a checkpoint for chatting.
    """

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # A training text already contains markers like <user> and <bot>, so the
    # model learns the shape of a conversation instead of raw disconnected text.
    texts = load_training_texts(
        dataset=args.dataset,
        corpus_dir=args.corpus_dir,
        max_pairs=args.max_pairs,
        hf_dataset_name=args.hf_dataset,
        hf_split=args.hf_split,
    )
    if not texts:
        raise ValueError("No training texts were loaded.")
    print(f"Loaded {len(texts):,} conversation pairs from {args.dataset}.")

    # The tokenizer decides which words/punctuation become known vocabulary.
    # Tokens outside the vocabulary become <unk>, which keeps the model small.
    tokenizer = SimpleTokenizer.build(
        texts,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
    )
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")

    # These settings define the model size. Increasing them usually improves
    # capacity but also makes training slower and more memory-heavy.
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_id,
    )

    # build_datasets creates x/y token pairs. x is the prompt tokens and y is
    # the same sequence shifted left, so y contains the next-token answers.
    datasets = build_datasets(
        texts,
        tokenizer,
        block_size=args.block_size,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    train_loader = DataLoader(
        datasets.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        datasets.valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TransformerChatModel(model_config).to(device)

    # AdamW is a common optimizer for Transformers. It updates model weights
    # using gradients while weight_decay gently discourages oversized weights.
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()
    step = 0
    last_train_loss = 0.0
    while step < args.steps:
        for x, y in train_loader:
            step += 1
            x = x.to(device)
            y = y.to(device)

            # Forward pass: get predictions and compare them with the correct
            # next tokens through the loss returned by the model.
            _, loss = model(x, y)

            # Backward pass: calculate gradients, clip extreme gradients, then
            # let the optimizer update the model parameters.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            last_train_loss = float(loss.item())
            if step % args.log_every == 0 or step == 1:
                print(f"step {step:>5}/{args.steps} | train loss {last_train_loss:.4f}")

            if step % args.eval_every == 0 or step == args.steps:
                valid_loss = estimate_loss(model, valid_loader, device)
                print(f"step {step:>5}/{args.steps} | valid loss {valid_loss:.4f}")

            if step >= args.steps:
                break

    checkpoint_path = os.path.join(args.output_dir, args.checkpoint_name)
    save_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        args,
        metrics={"train_loss": last_train_loss, "valid_loss": estimate_loss(model, valid_loader, device)},
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the training CLI."""

    parser = argparse.ArgumentParser(description="Train a small Transformer chatbot.")
    parser.add_argument("--dataset", choices=["cornell", "dailydialog"], default="cornell")
    parser.add_argument("--corpus-dir", default=os.path.join("data", "cornell movie-dialogs corpus"))
    parser.add_argument("--hf-dataset", default="OpenRL/daily_dialog")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--max-pairs", type=int, default=None)

    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--checkpoint-name", default="chatbot-small-llm.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available.")

    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-vocab-size", type=int, default=12000)
    parser.add_argument("--min-freq", type=int, default=2)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=200)
    return parser


def main() -> None:
    """CLI entrypoint used by train_llm.py."""

    args = build_arg_parser().parse_args()
    train(args)
