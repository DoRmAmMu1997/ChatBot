"""Tokenizer round-trip + chat template + tool parsing tests."""

from __future__ import annotations

import pytest

from chatbot.tokenizer.bpe import BPETokenizer, SpecialTokens
from chatbot.tokenizer.chat_template import default_chat_template, tool_chat_template
from chatbot.tokenizer.tool_template import (
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    format_tool_call,
    format_tool_result,
    parse_tool_calls,
)


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    text_dir = tmp_path_factory.mktemp("toktext")
    sample = text_dir / "sample.txt"
    sample.write_text(
        "Hello world. This is a tiny corpus used only to seed a BPE tokenizer for tests.\n"
        "We add a few sentences with punctuation, numbers like 42, and the word chatbot.",
        encoding="utf-8",
    )
    tk = BPETokenizer.train(files=[str(sample)], vocab_size=512, specials=SpecialTokens())
    return tk


def test_tokenizer_roundtrip(tiny_tokenizer):
    text = "Hello world!"
    ids = tiny_tokenizer.encode(text)
    decoded = tiny_tokenizer.decode(ids)
    assert decoded.strip().lower().startswith("hello"), decoded


def test_default_chat_template_renders():
    template = default_chat_template()
    rendered = template.render([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert "<|user|>" in rendered and "<|assistant|>" in rendered


def test_tool_call_parse_roundtrip():
    text = format_tool_call("read_file", {"path": "x.txt"})
    parsed = parse_tool_calls(text)
    assert len(parsed) == 1
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "x.txt"}


def test_tool_result_format():
    rendered = format_tool_result("read_file", {"content": "hi"})
    assert rendered.startswith("<|tool_result|>")
    assert "read_file" in rendered
