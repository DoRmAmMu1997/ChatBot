"""Chat templating: turn a list of messages into a single prompt string.

We use Jinja2 templates because they're the de facto standard in the
LLM ecosystem (Hugging Face tokenizers all carry a Jinja chat template).
Two flavours ship:

* ``default_chat_template()`` — plain ``<|system|>…<|user|>…<|assistant|>``
  rendering, used by both Aurora and Forge in non-tool mode.
* ``tool_chat_template()`` — Forge's tool-calling template, which renders
  tool definitions, tool calls, and tool results inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from jinja2 import Environment, StrictUndefined


@dataclass
class ChatTemplate:
    """A tiny wrapper around a Jinja string + the special tokens it uses."""

    template: str
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"

    def render(
        self,
        messages: Iterable[dict],
        *,
        add_generation_prompt: bool = True,
        tools: Optional[List[dict]] = None,
    ) -> str:
        env = Environment(undefined=StrictUndefined)
        compiled = env.from_string(self.template)
        return compiled.render(
            messages=list(messages),
            add_generation_prompt=add_generation_prompt,
            tools=tools or [],
            bos_token=self.bos_token,
            eos_token=self.eos_token,
        )


_DEFAULT_TEMPLATE = """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}{{ eos_token }}
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""


_TOOL_TEMPLATE = """{{ bos_token }}<|system|>
{% if messages and messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}
{% else %}You are a helpful coding assistant.{% endif %}
{% if tools %}
Available tools (JSON schema):
{% for tool in tools %}- {{ tool | tojson }}
{% endfor %}When you want to call a tool, emit a single
<|tool_call|>{"name":"…","arguments":{…}}<|/tool_call|>
block. Wait for the corresponding <|tool_result|>…<|/tool_result|> before continuing.
{% endif %}{{ eos_token }}
{% for message in messages %}{% if message['role'] == 'system' and loop.index0 == 0 %}{# already handled above #}{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'tool' %}<|tool_result|>
{{ message['content'] }}<|/tool_result|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>
{% endif %}"""


def default_chat_template() -> ChatTemplate:
    return ChatTemplate(_DEFAULT_TEMPLATE)


def tool_chat_template() -> ChatTemplate:
    return ChatTemplate(_TOOL_TEMPLATE)


def format_messages(
    messages: Iterable[dict],
    *,
    add_generation_prompt: bool = True,
    tools: Optional[List[dict]] = None,
    template: Optional[ChatTemplate] = None,
) -> str:
    """Render messages to a prompt string. Tools auto-switch to the tool template."""

    if template is None:
        template = tool_chat_template() if tools else default_chat_template()
    return template.render(messages, add_generation_prompt=add_generation_prompt, tools=tools)
