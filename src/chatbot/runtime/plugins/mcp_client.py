"""Minimal stdio MCP (Model Context Protocol) client.

MCP is an open protocol that lets language models talk to external tools
over a JSON-RPC channel. Servers expose tools, resources, and prompts;
clients (us, here) connect to a server, discover its tools, and bridge
them into the local tool registry so the model can call them like any
built-in tool.

We implement only what we need: stdio transport, JSON-RPC line-framed,
``initialize`` + ``tools/list`` + ``tools/call``. Network / SSE transports
are easy to add by swapping the read/write halves.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _PendingRequest:
    event: threading.Event
    response: Optional[Dict[str, Any]] = None


class MCPClient:
    """Live MCP connection over the stdio transport."""

    def __init__(self, command: List[str]):
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._next_id = 1
        self._pending: Dict[int, _PendingRequest] = {}
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._initialize()

    # -- public API ---------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()

    # -- internals ----------------------------------------------------

    def _initialize(self) -> None:
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "chatbot-runtime", "version": "1.0"},
        })

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            pending = _PendingRequest(event=threading.Event())
            self._pending[req_id] = pending
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }) + "\n"
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()
        pending.event.wait(timeout=60)
        with self._lock:
            self._pending.pop(req_id, None)
        if pending.response is None:
            raise TimeoutError(f"MCP request {method!r} timed out")
        return pending.response.get("result", {})

    def _read_loop(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            req_id = msg.get("id")
            if req_id is None:
                continue
            with self._lock:
                pending = self._pending.get(req_id)
            if pending is not None:
                pending.response = msg
                pending.event.set()
