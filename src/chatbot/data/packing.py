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

    # Running buffer. We append every source example here (followed by an
    # EOS) and emit a fixed-size chunk whenever the buffer is full enough.
    buffer: List[int] = []

    for ids in token_streams:
        # Drop empty examples — they'd just append a stray EOS with no
        # content and inflate the EOS frequency in training.
        if not ids:
            continue
        # Append the example tokens, then a single EOS to mark the
        # boundary. During training the model learns to "reset" at EOS
        # and start the next document.
        buffer.extend(ids)
        buffer.append(eos_token_id)

        # Drain as many full blocks as the buffer currently contains.
        # A typical batch will produce 0 or 1 blocks per source example;
        # the loop is here for the rare case where one example is bigger
        # than the block size and produces several.
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[block_size:]
            # ``labels`` is the same as ``input_ids`` because we use
            # next-token-prediction — the training loop shifts labels by
            # one internally.
            yield PackedExample(input_ids=chunk, labels=list(chunk))

    # Trailing partial block — pad to length, keep the labels matching.
    if buffer:
        padded = buffer + [pad_token_id] * (block_size - len(buffer))
        labels = buffer + [-100] * (block_size - len(buffer))
        yield PackedExample(input_ids=padded, labels=labels)
