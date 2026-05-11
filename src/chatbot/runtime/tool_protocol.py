"""Tool interface and registry.

A "tool" in this runtime is anything the agent can invoke: a function, an
MCP server method, a shell command. Every tool implements the same tiny
interface — a name, a JSON schema describing its arguments, and a callable
that takes those arguments and returns a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# A handler takes a dict of validated arguments and returns any
# JSON-serializable result. The runtime wraps the result in a
# ``<|tool_result|>`` block.
ToolHandler = Callable[[Dict[str, Any]], Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]            # JSON-schema fragment for arguments
    handler: ToolHandler
    plugin: Optional[str] = None           # which plugin owns this tool

    def to_schema(self) -> Dict[str, Any]:
        """Render the tool as a JSON-schema descriptor (for the prompt)."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """In-memory directory of registered tools.

    Lookup is O(1). Iteration order is insertion order so the system prompt
    rendering is stable across runs.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
