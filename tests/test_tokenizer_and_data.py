from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from src.chatbot.data import (
    load_jsonl_pairs,
    pair_to_training_text,
    pairs_from_messages,
    parse_dataset_names,
)
from src.chatbot.tokenizer import BOT_TOKEN, BPETokenizer, EOS_TOKEN, USER_TOKEN


def test_bpe_tokenizer_round_trips_save_and_load(tmp_path):
    texts = [
        "<bos> <user> hello there <bot> hi there <eos>",
        "<bos> <user> tokenizer test <bot> tokenizers split text <eos>",
    ]
    tokenizer = BPETokenizer.train(texts, vocab_size=64, min_frequency=1)
    path = tmp_path / "chatbot-bpe.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)
    ids = loaded.encode("hello tokenizer")

    assert ids
    assert loaded.decode(ids)
    assert loaded.vocab_size == tokenizer.vocab_size


def test_dataset_helpers_format_common_chat_text():
    pairs = load_jsonl_pairs(Path("tests/fixtures/sample_pairs.jsonl"))
    text = pair_to_training_text(pairs[0])

    assert len(pairs) == 3
    assert USER_TOKEN in text
    assert BOT_TOKEN in text
    assert EOS_TOKEN in text


def test_message_rows_convert_to_user_assistant_pairs():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]

    assert pairs_from_messages(messages) == [("Hi", "Hello")]


def test_dataset_name_parser_supports_mixed_and_lists():
    assert parse_dataset_names("cornell,dolly") == ["cornell", "dolly"]
    assert "ultrachat" in parse_dataset_names("mixed")
