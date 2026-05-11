"""Config loader tests — the ``defaults:`` chain in particular."""

from __future__ import annotations

from chatbot.utils.config import load_config


def test_load_tiny_model_config():
    cfg = load_config("models/tiny")
    assert cfg.family == "aurora"
    assert int(cfg.d_model) == 256


def test_runtime_defaults_chain():
    cfg = load_config("runtime/forge-coder")
    # forge-coder.yaml has `defaults: [default]`, so it should inherit
    # the system_prompt-shape from default.yaml and override what it
    # declares explicitly.
    assert "system_prompt" in cfg
    assert int(cfg.context_window) == 1_048_576
