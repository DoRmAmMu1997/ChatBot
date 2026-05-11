"""Filesystem tools: read, write, glob, grep.

Every tool respects the runtime's ``filesystem_allowlist``. Paths outside
the allowlist raise an ``OSError`` and the agent sees an error result —
which is exactly what we want, since the model shouldn't be silently
denied tool access without knowing.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..tool_protocol import Tool, ToolRegistry


def _allow(path: str | os.PathLike[str], allowlist: Optional[Sequence[str]]) -> Path:
    """Resolve ``path`` and refuse if it is outside every allowlisted prefix."""

    resolved = Path(path).expanduser().resolve()
    if not allowlist:
        return resolved
    for prefix in allowlist:
        expanded = Path(os.path.expandvars(prefix)).expanduser().resolve()
        # ``resolved`` is allowed if it equals an entry or sits under one.
        if resolved == expanded or expanded in resolved.parents:
            return resolved
    raise PermissionError(f"Path {resolved} is outside the filesystem allowlist.")


def register_filesystem_tools(
    registry: ToolRegistry,
    *,
    allowlist: Optional[Sequence[str]] = None,
) -> None:
    def _read_file(args: Dict[str, Any]) -> Dict[str, Any]:
        path = _allow(args["path"], allowlist)
        return {"content": path.read_text(encoding=args.get("encoding", "utf-8"))}

    def _write_file(args: Dict[str, Any]) -> Dict[str, Any]:
        path = _allow(args["path"], allowlist)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding=args.get("encoding", "utf-8"))
        return {"path": str(path), "bytes": len(args["content"])}

    def _glob(args: Dict[str, Any]) -> Dict[str, Any]:
        root = _allow(args.get("path", "."), allowlist)
        pattern = args["pattern"]
        matches: List[str] = []
        for p in root.rglob("*"):
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if fnmatch.fnmatch(str(rel), pattern):
                matches.append(str(p))
                if len(matches) >= int(args.get("limit", 1000)):
                    break
        return {"matches": matches}

    def _grep(args: Dict[str, Any]) -> Dict[str, Any]:
        root = _allow(args.get("path", "."), allowlist)
        regex = re.compile(args["pattern"])
        out: List[Dict[str, Any]] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    out.append({"path": str(p), "line": i, "text": line.rstrip()})
                    if len(out) >= int(args.get("limit", 500)):
                        return {"matches": out}
        return {"matches": out}

    registry.register(Tool(
        name="read_file",
        description="Read the entire contents of a UTF-8 text file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path"],
        },
        handler=_read_file,
        plugin="builtin.filesystem",
    ))
    registry.register(Tool(
        name="write_file",
        description="Write text to a file (overwrites). Creates parent dirs if missing.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
        plugin="builtin.filesystem",
    ))
    registry.register(Tool(
        name="glob",
        description="Recursive filename glob (fnmatch-style pattern).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "default": 1000},
            },
            "required": ["pattern"],
        },
        handler=_glob,
        plugin="builtin.filesystem",
    ))
    registry.register(Tool(
        name="grep",
        description="Recursive regex search across files in a directory tree.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string"},
                "limit": {"type": "integer", "default": 500},
            },
            "required": ["pattern"],
        },
        handler=_grep,
        plugin="builtin.filesystem",
    ))
