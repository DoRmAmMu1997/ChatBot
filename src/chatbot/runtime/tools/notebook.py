"""Stateful Python notebook tool.

A persistent ``exec`` namespace per agent run, so the model can build up
intermediate values across multiple tool calls — same as a Jupyter notebook
or Claude Code's notebook tool.
"""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict

from ..tool_protocol import Tool, ToolRegistry


class _Notebook:
    def __init__(self):
        self.globals: Dict[str, Any] = {"__name__": "__chatbot_notebook__"}

    def run(self, code: str) -> Dict[str, Any]:
        out_io, err_io = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out_io), redirect_stderr(err_io):
                exec(compile(code, "<notebook>", "exec"), self.globals)
            return {"stdout": out_io.getvalue(), "stderr": err_io.getvalue(), "ok": True}
        except Exception:
            tb = traceback.format_exc()
            return {
                "stdout": out_io.getvalue(),
                "stderr": err_io.getvalue() + tb,
                "ok": False,
            }


def register_notebook_tools(registry: ToolRegistry) -> None:
    nb = _Notebook()

    def _run(args: Dict[str, Any]) -> Dict[str, Any]:
        return nb.run(str(args["code"]))

    registry.register(Tool(
        name="python",
        description="Execute Python code in a persistent notebook session.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        handler=_run,
        plugin="builtin.notebook",
    ))
