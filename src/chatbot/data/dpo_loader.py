"""DPO data loader. Streams (prompt, chosen, rejected) token triples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import IterableDataset

from .registry import get_dataset


@dataclass
class DPOMixEntry:
    name: str
    weight: float


class DPODataset(IterableDataset):
    """Streams ``(prompt_ids, chosen_ids, rejected_ids)`` tensors.

    The DPO loss compares the policy's log-probability of the chosen reply
    vs. the rejected one (each conditioned on the same prompt). Padding is
    done per-batch to the longest sequence in the batch.
    """

    def __init__(
        self,
        mix: Sequence[DPOMixEntry],
        *,
        tokenizer,
        max_prompt_length: int,
        max_response_length: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.mix = list(mix)
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.pad_token_id = pad_token_id

    def _iter_pairs(self) -> Iterable[dict]:
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
                    yield next(it)
                    kept.append(it)
                except StopIteration:
                    pass
            iters = kept

    def __iter__(self) -> Iterator[dict]:
        for row in self._iter_pairs():
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)[
                : self.max_prompt_length
            ]
            chosen_ids = self.tokenizer.encode(chosen, add_special_tokens=False)[
                : self.max_response_length
            ]
            rejected_ids = self.tokenizer.encode(rejected, add_special_tokens=False)[
                : self.max_response_length
            ]
            yield {
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
                "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            }


def _pad_right(tensors: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Pad a list of 1-D tensors on the right to the longest length."""

    max_len = max(t.numel() for t in tensors)
    out = torch.full((len(tensors), max_len), pad_id, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t
    return out


def collate_dpo_batch(batch: List[dict], *, pad_id: int) -> dict:
    """Stack a list of triples into batched right-padded tensors."""

    return {
        "prompt_ids": _pad_right([b["prompt_ids"] for b in batch], pad_id),
        "chosen_ids": _pad_right([b["chosen_ids"] for b in batch], pad_id),
        "rejected_ids": _pad_right([b["rejected_ids"] for b in batch], pad_id),
    }
