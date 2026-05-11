"""Tool-use SFT loader. Same shape as SFT, but masks tool-result text.

Why mask tool-result text: the model isn't supposed to *predict* the
content of a tool's response — those tokens come from the environment.
We only want loss on the model's *decisions to call tools* and on its
final natural-language answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import IterableDataset

from ..tokenizer.chat_template import tool_chat_template
from ..tokenizer.tool_template import TOOL_RESULT_CLOSE, TOOL_RESULT_OPEN
from .registry import get_dataset


@dataclass
class ToolMixEntry:
    name: str
    weight: float


class ToolUseDataset(IterableDataset):
    """Streams ``(input_ids, labels)`` for tool-use SFT."""

    def __init__(
        self,
        mix: Sequence[ToolMixEntry],
        *,
        tokenizer,
        block_size: int,
        pad_token_id: int,
        mask_tool_results: bool = True,
    ):
        super().__init__()
        self.mix = list(mix)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.mask_tool_results = mask_tool_results
        self.template = tool_chat_template()

    def _iter_chats(self) -> Iterable[List[dict]]:
        iters = []
        for entry in self.mix:
            try:
                spec = get_dataset(entry.name)
                iters.append(iter(spec.loader()))
            except Exception:
                continue
        while iters:
            kept = []
            for it in iters:
                try:
                    row = next(it)
                    msgs = row.get("messages") or []
                    if msgs:
                        yield msgs
                    kept.append(it)
                except StopIteration:
                    pass
            iters = kept

    def __iter__(self) -> Iterator[dict]:
        for messages in self._iter_chats():
            rendered = self.template.render(messages, add_generation_prompt=False)
            ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            labels = list(ids)
            if self.mask_tool_results:
                labels = self._mask_tool_results(rendered, ids, labels)
            if len(ids) > self.block_size:
                ids = ids[: self.block_size]
                labels = labels[: self.block_size]
            else:
                pad = self.block_size - len(ids)
                ids = ids + [self.pad_token_id] * pad
                labels = labels + [-100] * pad
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    def _mask_tool_results(self, rendered: str, ids: List[int], labels: List[int]) -> List[int]:
        """Replace tokens that fall inside ``<|tool_result|>...<|/tool_result|>`` with -100."""

        char_to_token = self._char_to_token_map(rendered, ids)
        cursor = 0
        while True:
            start = rendered.find(TOOL_RESULT_OPEN, cursor)
            if start == -1:
                break
            end = rendered.find(TOOL_RESULT_CLOSE, start)
            if end == -1:
                break
            end += len(TOOL_RESULT_CLOSE)
            tok_start = char_to_token.get(start, 0)
            tok_end = char_to_token.get(end, len(labels))
            for i in range(tok_start, tok_end):
                if 0 <= i < len(labels):
                    labels[i] = -100
            cursor = end
        return labels

    def _char_to_token_map(self, text: str, ids: List[int]) -> dict:
        """Approximate char-to-token map by re-encoding prefixes.

        Not the fastest method, but readable. For production, your tokenizer
        library will expose offsets directly — swap that in here if speed
        matters.
        """

        cumulative: dict = {0: 0}
        decoded_chars = 0
        for tok_idx in range(1, len(ids) + 1):
            decoded = self.tokenizer.decode(ids[:tok_idx], skip_special_tokens=False)
            cumulative[len(decoded)] = tok_idx
            decoded_chars = len(decoded)
            if decoded_chars >= len(text):
                break
        return cumulative
