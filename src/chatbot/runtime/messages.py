"""Message and content-block data types used by the agent loop.

We model conversations as a list of :class:`AgentMessage`. Each message has
a role (``"system"``, ``"user"``, ``"assistant"``, ``"tool"``) and a list of
content blocks. A content block is either plain text, a tool call (when
emitted by the assistant), or a tool result (when produced by a tool).

Keeping content as structured blocks — instead of a single string — makes
it easy for the runtime to know exactly what came from where, and lets
hooks intercept individual block types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ContentBlock:
    """One piece of content inside a message."""

    type: Literal["text", "tool_call", "tool_result", "image"]
    # For text/tool_result.
    text: Optional[str] = None
    # For tool_call.
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    # For tool_result.
    tool_call_id: Optional[str] = None
    is_error: bool = False
    # For image (Aurora).
    image_ref: Optional[str] = None


@dataclass
class AgentMessage:
    role: Role
    content: List[ContentBlock] = field(default_factory=list)

    def text(self) -> str:
        """Return concatenated text from all text-type blocks."""

        return "\n".join(b.text for b in self.content if b.type == "text" and b.text)

    @classmethod
    def from_string(cls, role: Role, text: str) -> "AgentMessage":
        return cls(role=role, content=[ContentBlock(type="text", text=text)])
