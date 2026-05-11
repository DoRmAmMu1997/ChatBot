"""Parsing and rendering of tool calls inside model output.

The contract:

  * The model emits ``<|tool_call|>{"name": ..., "arguments": {...}}<|/tool_call|>``.
  * The runtime parses each block, dispatches the named tool, then
    pastes the result back into the prompt wrapped in
    ``<|tool_result|>...<|/tool_result|>``.

We keep parsing tolerant: extra whitespace, comma-trailing JSON, and
multiple back-to-back tool calls all parse cleanly. Validation against the
expected JSON-schema is left to the runtime tool dispatcher.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


TOOL_CALL_OPEN = "<|tool_call|>"
TOOL_CALL_CLOSE = "<|/tool_call|>"
TOOL_RESULT_OPEN = "<|tool_result|>"
TOOL_RESULT_CLOSE = "<|/tool_result|>"

_TOOL_CALL_PATTERN = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw: str


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Extract every ``<|tool_call|>…<|/tool_call|>`` block from a string."""

    results: List[ToolCall] = []
    for match in _TOOL_CALL_PATTERN.finditer(text):
        body = match.group(1).strip()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            # Be tolerant — strip trailing commas, retry once.
            cleaned = re.sub(r",\s*([}\]])", r"\1", body)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        if not isinstance(parsed, dict):
            continue
        name = str(parsed.get("name", "")).strip()
        args = parsed.get("arguments", {}) or {}
        if not isinstance(args, dict):
            continue
        if not name:
            continue
        results.append(ToolCall(name=name, arguments=args, raw=match.group(0)))
    return results


def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Render a tool call in the format the model is trained to emit."""

    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"


def format_tool_result(name: str, result: Any) -> str:
    """Render a tool result block for re-injection into the prompt."""

    if not isinstance(result, str):
        try:
            result = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            result = str(result)
    return f"{TOOL_RESULT_OPEN}{name}: {result}{TOOL_RESULT_CLOSE}"


def strip_partial_tool_call(text: str) -> str:
    """Remove an unfinished ``<|tool_call|>...`` suffix from a streaming chunk.

    Useful when a streamed model output ends mid-tool-call — we hide the
    fragment until we have the complete block.
    """

    idx = text.rfind(TOOL_CALL_OPEN)
    if idx == -1:
        return text
    if TOOL_CALL_CLOSE in text[idx:]:
        return text
    return text[:idx]
