"""SFT data loader. Renders messages with the chat template and masks user turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import IterableDataset

from ..tokenizer.chat_template import default_chat_template, tool_chat_template
from .registry import get_dataset


@dataclass
class SFTMixEntry:
    name: str
    weight: float


class SFTDataset(IterableDataset):
    """Streams ``(input_ids, labels)`` for supervised fine-tuning.

    The label tensor is the same as ``input_ids`` *except* every position
    that came from a user / system / tool turn is replaced with ``-100``.
    PyTorch's cross-entropy ignores ``-100``, so the loss is computed only
    on assistant-generated tokens.
    """

    def __init__(
        self,
        mix: Sequence[SFTMixEntry],
        *,
        tokenizer,
        block_size: int,
        pad_token_id: int,
        mask_user_turns: bool = True,
        use_tool_template: bool = False,
    ):
        super().__init__()
        self.mix = list(mix)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.mask_user_turns = mask_user_turns
        self.template = tool_chat_template() if use_tool_template else default_chat_template()

    def _iter_chats(self) -> Iterable[List[dict]]:
        iters = []
        for entry in self.mix:
            try:
                spec = get_dataset(entry.name)
                iters.append((entry.weight, iter(spec.loader())))
            except Exception:
                continue
        round_idx = 0
        while iters:
            kept = []
            for weight, it in iters:
                try:
                    row = next(it)
                    messages = row.get("messages") or []
                    if messages:
                        yield messages
                    kept.append((weight, it))
                except StopIteration:
                    pass
            iters = kept
            round_idx += 1

    def __iter__(self) -> Iterator[dict]:
        for messages in self._iter_chats():
            # Render the whole conversation, then re-encode each turn so we
            # know where the assistant tokens are. We do this in two passes:
            # one to assemble the full token sequence, one to find which
            # spans are assistant turns. This is the most readable approach,
            # though not the fastest.
            full = self.template.render(messages, add_generation_prompt=False)
            full_ids = self.tokenizer.encode(full, add_special_tokens=False)

            if not self.mask_user_turns:
                labels = list(full_ids)
            else:
                labels = [-100] * len(full_ids)
                # For each assistant turn, find its substring in the rendered
                # text and mark the corresponding token span as a label.
                cursor = 0
                for msg in messages:
                    rendered_turn = self.template.render([msg], add_generation_prompt=False)
                    turn_ids = self.tokenizer.encode(rendered_turn, add_special_tokens=False)
                    span_len = len(turn_ids)
                    if msg.get("role") == "assistant":
                        labels[cursor : cursor + span_len] = full_ids[cursor : cursor + span_len]
                    cursor += span_len

            # Truncate / pad to block_size.
            if len(full_ids) > self.block_size:
                full_ids = full_ids[: self.block_size]
                labels = labels[: self.block_size]
            else:
                pad_amount = self.block_size - len(full_ids)
                full_ids = full_ids + [self.pad_token_id] * pad_amount
                labels = labels + [-100] * pad_amount

            yield {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


def collate_sft_batch(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
    }
