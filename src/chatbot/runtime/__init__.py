"""Claude-Code-style runtime: tools, plugins, skills, hooks, slash commands.

This package wraps a generation backend (the language model) with an
agent loop that:

* parses ``<|tool_call|>`` blocks from the model's output,
* dispatches them through registered tools (filesystem, shell, etc.),
* feeds the tool result back as a ``<|tool_result|>`` message,
* respects pre/post hooks and skill-injection rules.

Most users will only touch :func:`build_agent`. Everything else is
extensible for users who want their own tools or plugins.
"""

from .agent import Agent, build_agent
from .messages import AgentMessage, ContentBlock
from .tool_protocol import Tool, ToolRegistry

__all__ = ["Agent", "build_agent", "AgentMessage", "ContentBlock", "Tool", "ToolRegistry"]
