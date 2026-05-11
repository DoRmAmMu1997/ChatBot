"""Smoke tests for the data-loading utilities — no network required."""

from __future__ import annotations

import torch

from chatbot.data.packing import pack_sequences


def test_packing_emits_uniform_blocks():
    streams = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    out = list(pack_sequences(streams, block_size=4, eos_token_id=0))
    for example in out:
        assert len(example.input_ids) == 4
        assert len(example.labels) == 4


def test_packing_inserts_eos_between_streams():
    streams = [[1, 2], [3, 4]]
    out = list(pack_sequences(streams, block_size=6, eos_token_id=99))
    # Stream A → 1,2,99  Stream B → 3,4,99  Final block: [1,2,99,3,4,99].
    assert out[0].input_ids == [1, 2, 99, 3, 4, 99]
