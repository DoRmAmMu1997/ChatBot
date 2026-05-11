"""DataLoader for pretraining (next-token prediction on a mix of corpora)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import torch
from torch.utils.data import IterableDataset

from .packing import pack_sequences
from .registry import DatasetSpec, get_dataset


@dataclass
class MixEntry:
    name: str
    weight: float


class PretrainDataset(IterableDataset):
    """Streams (input_ids, labels) tensors from a weighted mix of datasets.

    The underlying loaders are Hugging Face streaming datasets, so this works
    on corpora much larger than RAM. Token ids come from a BPE tokenizer
    passed in by the training script.
    """

    def __init__(
        self,
        mix: Sequence[MixEntry],
        *,
        tokenizer,
        block_size: int,
        eos_token_id: int,
        pad_token_id: int,
        shuffle_buffer: int = 100000,
        seed: int = 42,
    ):
        super().__init__()
        self.mix = list(mix)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _iter_text_streams(self) -> Iterable[str]:
        """Yield raw text strings sampled from each mix component by weight.

        This is a *very* simple sampler — it just round-robins by integer
        weight share. For frontier training runs you'd use a proper
        temperature-mixed iterator; round-robin is enough for the smoke test
        + a clear starting point for users.
        """

        iters = []
        for entry in self.mix:
            try:
                spec: DatasetSpec = get_dataset(entry.name)
                iters.append((entry.weight, iter(spec.loader())))
            except Exception:
                # Be tolerant — skip a dataset that the user hasn't downloaded.
                continue
        if not iters:
            return

        # Integer "ticket" count per entry, scaled so the smallest weight = 1.
        weights = [w for w, _ in iters]
        min_w = min(weights) if weights else 1.0
        tickets = [max(1, int(round(w / min_w))) for w in weights]

        round_idx = 0
        while iters:
            kept = []
            for (weight, it), ticket in zip(iters, tickets):
                for _ in range(ticket):
                    try:
                        row = next(it)
                        text = row.get("text") or row.get("content")
                        if text:
                            yield text
                    except StopIteration:
                        break
                else:
                    kept.append((weight, it))
                    continue
                # ``break`` from the inner for-loop: this iterator is exhausted.
                continue
            iters = kept
            round_idx += 1

    def __iter__(self) -> Iterator[dict]:
        text_stream = self._iter_text_streams()

        def token_streams():
            for text in text_stream:
                # encode without adding specials — packing handles EOS.
                yield self.tokenizer.encode(text, add_special_tokens=False)

        for packed in pack_sequences(
            token_streams(),
            block_size=self.block_size + 1,   # +1 so we can shift for labels
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        ):
            input_ids = torch.tensor(packed.input_ids[:-1], dtype=torch.long)
            labels = torch.tensor(packed.labels[1:], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


def collate_pretrain_batch(batch: List[dict]) -> dict:
    """Stack a list of single examples into a batch tensor dict."""

    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}
