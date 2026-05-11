"""Pack many short sequences end-to-end into fixed-length training blocks.

A naive batch wastes a lot of GPU time on padding when sequences are
variable-length. Packing concatenates many examples back-to-back, separated
by EOS markers, and slices the result into uniform ``block_size`` chunks.
A small amount of "context bleed" across document boundaries is the cost,
but the throughput win is enormous (often 2-4x).

This module is deliberately framework-light: it works on plain lists of
token ids and yields plain dicts, so it can be used by any data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List


@dataclass
class PackedExample:
    input_ids: List[int]
    labels: List[int]


def pack_sequences(
    token_streams: Iterable[List[int]],
    *,
    block_size: int,
    eos_token_id: int,
    pad_token_id: int = 0,
) -> Iterator[PackedExample]:
    """Yield fixed-length examples assembled from a stream of variable-length token lists.

    Args:
        token_streams: iterable of pre-tokenized examples (lists of ints).
        block_size: target length of each emitted example.
        eos_token_id: id appended after each source example to separate them.
        pad_token_id: id used to fill the final partial block.
    """

    buffer: List[int] = []

    for ids in token_streams:
        # Drop empty examples — they'd just append a stray EOS.
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos_token_id)

        # Drain as many full blocks as possible.
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[block_size:]
            yield PackedExample(input_ids=chunk, labels=list(chunk))

    # Trailing partial block — pad to length, keep the labels matching.
    if buffer:
        padded = buffer + [pad_token_id] * (block_size - len(buffer))
        labels = buffer + [-100] * (block_size - len(buffer))
        yield PackedExample(input_ids=padded, labels=labels)
