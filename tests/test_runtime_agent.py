"""Smoke tests for the agent loop using a tiny stub model."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from chatbot.runtime.agent import Agent, build_agent
from chatbot.tokenizer.chat_template import format_messages


class _StubModel(nn.Module):
    """Pretends to be a model. Always 'generates' a fixed reply."""

    def __init__(self, reply: str, tokenizer):
        super().__init__()
        self.reply = reply
        self.tokenizer = tokenizer
        # The runtime checks ``next(model.parameters()).device`` and
        # ``model.config.vision.num_image_tokens`` — give it both.
        self._dummy = nn.Parameter(torch.zeros(1))

        class _Cfg:
            class _V:
                num_image_tokens = 0
            vision = _V()
        self.config = _Cfg()

    def generate(self, input_ids, **kwargs):
        suffix = self.tokenizer.encode(self.reply, add_special_tokens=False)
        return torch.tensor([input_ids[0].tolist() + suffix], dtype=torch.long)


def _runtime_config(**overrides):
    cfg = OmegaConf.create({
        "system_prompt": "test prompt",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_new_tokens": 8,
        "tools_enabled": False,
        "skills_enabled": True,
        "skills_auto_select": True,
        "skills_max_active": 4,
        "hooks_enabled": True,
        "slash_commands_enabled": True,
        "subagents_enabled": False,
        "subagent_max_concurrency": 1,
        "enabled_plugins": [],
        "extra_plugin_paths": [],
        "plugins_dir": "/nonexistent",
        "safety": {},
        "tool_max_iterations": 4,
    })
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def test_agent_responds_without_tools(tmp_path):
    from chatbot.tokenizer.bpe import BPETokenizer, SpecialTokens

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world test agent runtime", encoding="utf-8")
    tokenizer = BPETokenizer.train(files=[str(text_file)], vocab_size=256, specials=SpecialTokens())
    model = _StubModel("hi there", tokenizer)
    cfg = _runtime_config()
    agent = build_agent(model=model, tokenizer=tokenizer, runtime_cfg=cfg)
    reply = agent.respond("hello")
    assert isinstance(reply, str)


def test_slash_command_help(tmp_path):
    from chatbot.tokenizer.bpe import BPETokenizer, SpecialTokens

    text_file = tmp_path / "corpus.txt"
    text_file.write_text("hello world", encoding="utf-8")
    tokenizer = BPETokenizer.train(files=[str(text_file)], vocab_size=256, specials=SpecialTokens())
    model = _StubModel("ignored", tokenizer)
    cfg = _runtime_config()
    agent = build_agent(model=model, tokenizer=tokenizer, runtime_cfg=cfg)
    # /help is registered automatically — resolving it should NOT call the model.
    resolved = agent.slash_registry.resolve("/help")
    assert resolved is not None
    assert "Commands" in resolved
