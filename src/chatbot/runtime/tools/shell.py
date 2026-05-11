"""Shell tool — runs a subprocess with an allowlist + timeout.

Security note: we do not run arbitrary shell strings through ``shell=True``.
Each invocation explicitly lists the command tokens; the model is expected
to emit a ``cmd`` array, not a shell line. The first token must be in the
allowlist.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from ..tool_protocol import Tool, ToolRegistry


def register_shell_tools(
    registry: ToolRegistry,
    *,
    allowlist: Optional[Sequence[str]] = None,
    default_timeout: int = 60,
    cwd: Optional[str] = None,
) -> None:
    allowed = set(allowlist or [])

    def _run(args: Dict[str, Any]) -> Dict[str, Any]:
        cmd: List[str] = list(args.get("cmd", []))
        if not cmd:
            raise ValueError("Tool 'shell' requires a non-empty 'cmd' array.")
        head = os.path.basename(cmd[0])
        if allowed and head not in allowed:
            raise PermissionError(
                f"Shell command {head!r} is not in the runtime allowlist."
            )
        timeout = int(args.get("timeout", default_timeout))
        proc = subprocess.run(
            cmd,
            cwd=args.get("cwd", cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }

    registry.register(Tool(
        name="shell",
        description=(
            "Run a shell command. Args: cmd (array of strings), cwd (string), "
            "timeout (int seconds). Returns stdout / stderr / returncode."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "timeout": {"type": "integer", "default": default_timeout},
            },
            "required": ["cmd"],
        },
        handler=_run,
        plugin="builtin.shell",
    ))
