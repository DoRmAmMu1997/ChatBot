"""Dataset loading and PyTorch dataset helpers.

This module is responsible for turning human-readable conversations into text
strings the language model can learn from. Cornell is the zero-setup default,
and DailyDialog is available as an optional cleaner everyday-conversation
dataset when Hugging Face datasets is installed.
"""

from __future__ import annotations

import ast
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, random_split

from .tokenizer import BOS_TOKEN, BOT_TOKEN, EOS_TOKEN, PAD_TOKEN, USER_TOKEN


# A DialogPair is one prompt and the next response in the conversation.
DialogPair = Tuple[str, str]


@dataclass(frozen=True)
class DatasetBundle:
    """Container for train and validation datasets."""

    train: Dataset
    valid: Dataset


def load_cornell_pairs(corpus_dir: str, max_pairs: int | None = None) -> List[DialogPair]:
    """Load question/answer pairs from the Cornell Movie Dialogues corpus.

    Cornell stores the actual lines and the conversation order in separate
    files. We first build a line-id lookup, then walk through each conversation
    and pair every line with the line that follows it.
    """

    lines_path = os.path.join(corpus_dir, "movie_lines.txt")
    conversations_path = os.path.join(corpus_dir, "movie_conversations.txt")
    if not os.path.exists(lines_path) or not os.path.exists(conversations_path):
        raise FileNotFoundError(
            "Cornell files were not found. Expected movie_lines.txt and "
            f"movie_conversations.txt under: {corpus_dir}"
        )

    # movie_lines.txt maps a line id like L1045 to the actual spoken text. This
    # dictionary lets us quickly turn conversation ids into real words.
    id_to_text = {}
    with open(lines_path, "r", encoding="iso-8859-1") as file:
        for raw_line in file:
            parts = raw_line.rstrip("\n").split(" +++$+++ ")
            if len(parts) >= 5:
                line_id = parts[0]
                text = parts[4]
                id_to_text[line_id] = text

    pairs: List[DialogPair] = []
    with open(conversations_path, "r", encoding="iso-8859-1") as file:
        for raw_line in file:
            parts = raw_line.rstrip("\n").split(" +++$+++ ")
            if len(parts) < 4:
                continue

            # The final field looks like "['L1', 'L2']". ast.literal_eval reads
            # that Python-style list safely without executing code.
            try:
                utterance_ids = ast.literal_eval(parts[3])
            except (SyntaxError, ValueError):
                continue

            # Adjacent turns become training examples:
            # line A is the user prompt, and line B is the bot response.
            for left_id, right_id in zip(utterance_ids, utterance_ids[1:]):
                prompt = id_to_text.get(left_id, "").strip()
                response = id_to_text.get(right_id, "").strip()
                if prompt and response:
                    pairs.append((prompt, response))
                    if max_pairs is not None and len(pairs) >= max_pairs:
                        return pairs

    return pairs


def load_dailydialog_pairs(
    dataset_name: str = "OpenRL/daily_dialog",
    split: str = "train",
    max_pairs: int | None = None,
) -> List[DialogPair]:
    """Load DailyDialog adjacent-turn pairs from Hugging Face datasets.

    DailyDialog is optional because it requires an internet download and the
    extra datasets package. The rest of the project works without it.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "DailyDialog loading needs the optional 'datasets' package. "
            "Install requirements.txt or use --dataset cornell."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    pairs: List[DialogPair] = []
    for row in dataset:
        turns = row.get("dialog", [])
        # Like Cornell, every neighboring pair of turns becomes one example.
        for prompt, response in zip(turns, turns[1:]):
            if prompt and response:
                pairs.append((prompt, response))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs

    return pairs


def pair_to_training_text(pair: DialogPair) -> str:
    """Format one dialog pair as a single autoregressive training sequence.

    Autoregressive training means the model reads one long sequence and learns
    to predict each next token. Special markers teach it where the user message
    ends and where the bot answer begins.
    """

    prompt, response = pair
    return f"{BOS_TOKEN} {USER_TOKEN} {prompt} {BOT_TOKEN} {response} {EOS_TOKEN}"


def load_training_texts(
    dataset: str,
    corpus_dir: str,
    max_pairs: int | None,
    hf_dataset_name: str,
    hf_split: str,
) -> List[str]:
    """Load dialog pairs and format them for language-model training."""

    if dataset == "cornell":
        pairs = load_cornell_pairs(corpus_dir, max_pairs=max_pairs)
    elif dataset == "dailydialog":
        pairs = load_dailydialog_pairs(
            dataset_name=hf_dataset_name,
            split=hf_split,
            max_pairs=max_pairs,
        )
    else:
        raise ValueError("dataset must be either 'cornell' or 'dailydialog'")

    return [pair_to_training_text(pair) for pair in pairs]


class ConversationDataset(Dataset):
    """Turns formatted text into next-token-prediction examples.

    For each example, x is the input tokens and y is the same sequence shifted
    one token to the left. The model learns to predict y from x.
    """

    def __init__(self, texts: Sequence[str], tokenizer, block_size: int):
        self.pad_id = tokenizer.token_to_id[PAD_TOKEN]
        self.examples: List[torch.Tensor] = []

        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) < 2:
                continue

            # We need block_size + 1 ids because the target is shifted by one.
            # Example:
            # ids = [<bos>, <user>, hello, <bot>, hi]
            # x   = [<bos>, <user>, hello, <bot>]
            # y   = [<user>, hello, <bot>, hi]
            ids = ids[: block_size + 1]
            if len(ids) < block_size + 1:
                # Padding makes every example the same length, which allows
                # PyTorch to stack many examples into one batch tensor.
                ids = ids + [self.pad_id] * (block_size + 1 - len(ids))
            self.examples.append(torch.tensor(ids, dtype=torch.long))

        if not self.examples:
            raise ValueError("No usable training examples were produced.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        ids = self.examples[index]
        # Return input tokens and the correct next-token targets.
        return ids[:-1], ids[1:]


def build_datasets(
    texts: Sequence[str],
    tokenizer,
    block_size: int,
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetBundle:
    """Create reproducible train/validation splits.

    The training split updates the model. The validation split is held back so
    we can check whether the model is learning beyond memorizing one batch.
    """

    full_dataset = ConversationDataset(texts, tokenizer, block_size)
    valid_size = max(1, int(len(full_dataset) * valid_fraction))
    train_size = len(full_dataset) - valid_size
    if train_size <= 0:
        raise ValueError("Need at least two examples to create a validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(
        full_dataset,
        [train_size, valid_size],
        generator=generator,
    )
    return DatasetBundle(train=train_dataset, valid=valid_dataset)


def sample_texts(texts: Sequence[str], count: int, seed: int) -> List[str]:
    """Return a deterministic sample for quick experiments."""

    if count >= len(texts):
        return list(texts)
    rng = random.Random(seed)
    return rng.sample(list(texts), count)
