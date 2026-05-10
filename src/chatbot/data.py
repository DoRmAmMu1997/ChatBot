"""Dataset loading and PyTorch dataset helpers.

This module is responsible for turning human-readable conversations into text
strings the language model can learn from. Cornell is the zero-setup default,
and DailyDialog is available as an optional cleaner everyday-conversation
dataset when Hugging Face datasets is installed.
"""

from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, random_split

from .tokenizer import BOS_TOKEN, BOT_TOKEN, EOS_TOKEN, PAD_TOKEN, USER_TOKEN


# A DialogPair is one prompt and the next response in the conversation.
DialogPair = Tuple[str, str]


DATASET_RECIPES: Dict[str, Dict[str, str]] = {
    "cornell": {
        "type": "local",
        "description": "Bundled Cornell Movie Dialogues adjacent turns.",
    },
    "dailydialog": {
        "type": "huggingface",
        "dataset": "OpenRL/daily_dialog",
        "description": "Short daily-life conversations.",
    },
    "ultrachat": {
        "type": "huggingface",
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "description": "Instruction-style user/assistant chat data.",
    },
    "oasst1": {
        "type": "huggingface",
        "dataset": "OpenAssistant/oasst1",
        "description": "OpenAssistant conversation tree prompt/assistant pairs.",
    },
    "dolly": {
        "type": "huggingface",
        "dataset": "databricks/databricks-dolly-15k",
        "description": "Instruction, optional context, and response examples.",
    },
}


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


def _load_hf_dataset(dataset_name: str, split: str):
    """Import and load a Hugging Face dataset only when it is requested."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading needs the optional 'datasets' package. "
            "Install requirements.txt or use --dataset cornell."
        ) from exc

    # Keeping this in a tiny helper means importing src.chatbot.data does not
    # require downloading or even installing Hugging Face datasets.
    return load_dataset(dataset_name, split=split)


def pairs_from_messages(messages: Sequence[dict]) -> List[DialogPair]:
    """Convert chat message dictionaries into user/assistant pairs."""

    pairs: List[DialogPair] = []
    for left, right in zip(messages, messages[1:]):
        left_role = str(left.get("role", "")).lower()
        right_role = str(right.get("role", "")).lower()

        # Most chat datasets store a list like:
        # [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
        # We only keep pairs where the direction is user -> assistant.
        if left_role in {"user", "prompter"} and right_role in {"assistant", "bot"}:
            prompt = str(left.get("content") or left.get("text") or "").strip()
            response = str(right.get("content") or right.get("text") or "").strip()
            if prompt and response:
                pairs.append((prompt, response))
    return pairs


def load_ultrachat_pairs(
    dataset_name: str = DATASET_RECIPES["ultrachat"]["dataset"],
    split: str = "train_sft",
    max_pairs: int | None = None,
) -> List[DialogPair]:
    """Load UltraChat user/assistant pairs from its messages column."""

    dataset = _load_hf_dataset(dataset_name, split=split)
    pairs: List[DialogPair] = []
    for row in dataset:
        for pair in pairs_from_messages(row.get("messages", [])):
            pairs.append(pair)
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def load_oasst1_pairs(
    dataset_name: str = DATASET_RECIPES["oasst1"]["dataset"],
    split: str = "train",
    max_pairs: int | None = None,
) -> List[DialogPair]:
    """Load English OpenAssistant prompt/assistant pairs.

    OASST1 is a conversation tree. Each assistant row points at its parent
    prompt, so we build a lookup and pair assistant messages with their parent.
    """

    dataset = _load_hf_dataset(dataset_name, split=split)
    rows = [row for row in dataset if row.get("lang") in {None, "en"}]
    by_id = {row.get("message_id"): row for row in rows}
    pairs: List[DialogPair] = []

    for row in rows:
        if row.get("role") != "assistant":
            continue
        parent = by_id.get(row.get("parent_id"))
        if not parent or parent.get("role") not in {"prompter", "user"}:
            continue
        prompt = str(parent.get("text", "")).strip()
        response = str(row.get("text", "")).strip()
        if prompt and response:
            pairs.append((prompt, response))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def load_dolly_pairs(
    dataset_name: str = DATASET_RECIPES["dolly"]["dataset"],
    split: str = "train",
    max_pairs: int | None = None,
) -> List[DialogPair]:
    """Load Databricks Dolly instruction/response rows as dialog pairs."""

    dataset = _load_hf_dataset(dataset_name, split=split)
    pairs: List[DialogPair] = []
    for row in dataset:
        instruction = str(row.get("instruction", "")).strip()
        context = str(row.get("context", "")).strip()
        response = str(row.get("response", "")).strip()
        if context:
            instruction = f"{instruction}\n\nContext:\n{context}"
        if instruction and response:
            pairs.append((instruction, response))
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

    dataset_names = parse_dataset_names(dataset)
    pairs: List[DialogPair] = []
    for dataset_name in dataset_names:
        # Every loader returns the same simple shape: (prompt, response). That
        # common shape lets the training code stay independent of dataset quirks.
        if dataset_name == "cornell":
            pairs.extend(load_cornell_pairs(corpus_dir, max_pairs=max_pairs))
        elif dataset_name == "dailydialog":
            pairs.extend(
                load_dailydialog_pairs(
                    dataset_name=hf_dataset_name if dataset == "dailydialog" else DATASET_RECIPES["dailydialog"]["dataset"],
                    split=hf_split,
                    max_pairs=max_pairs,
                )
            )
        elif dataset_name == "ultrachat":
            pairs.extend(load_ultrachat_pairs(split=hf_split, max_pairs=max_pairs))
        elif dataset_name == "oasst1":
            pairs.extend(load_oasst1_pairs(split=hf_split, max_pairs=max_pairs))
        elif dataset_name == "dolly":
            pairs.extend(load_dolly_pairs(split=hf_split, max_pairs=max_pairs))
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    return [pair_to_training_text(pair) for pair in pairs]


def parse_dataset_names(dataset: str) -> List[str]:
    """Parse one dataset name, a comma-separated list, or ``mixed``."""

    if dataset == "mixed":
        return ["cornell", "dailydialog", "ultrachat", "oasst1", "dolly"]
    names = [name.strip().lower() for name in dataset.split(",") if name.strip()]
    return names or ["cornell"]


def write_dataset_manifest(path: str | Path) -> None:
    """Write dataset recipe metadata for readers and training scripts."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The manifest documents what data can be loaded without checking Python
    # code. It is intentionally metadata only, not the full external datasets.
    path.write_text(json.dumps(DATASET_RECIPES, indent=2), encoding="utf-8")


def load_jsonl_pairs(path: str | Path) -> List[DialogPair]:
    """Load local JSONL fixtures with ``prompt`` and ``response`` fields."""

    pairs: List[DialogPair] = []
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            row = json.loads(raw_line)
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if prompt and response:
                pairs.append((prompt, response))
    return pairs


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

    # A fixed generator seed makes the split repeatable. That is useful when
    # comparing two training runs because they validate on the same examples.
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
