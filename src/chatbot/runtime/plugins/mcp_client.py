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
    """Live MCP connection over the stdio transport.

    Threading model:

    * The MCP server runs as a *subprocess*; we talk to it over its
      stdin / stdout.
    * A background reader thread consumes lines from the server's stdout
      and matches each one to a pending request by ``id``.
    * The main thread sends a request, then blocks on a per-request
      ``threading.Event`` until the reader thread fills in the response.

    A ``_lock`` guards ``_next_id`` and ``_pending`` so concurrent calls
    from multiple agent threads don't trample each other.
    """

    def __init__(self, command: List[str]):
        # Spawn the MCP server. ``text=True`` lets us read/write strings
        # rather than bytes — MCP is line-framed JSON so strings are fine.
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # JSON-RPC request id counter. Starts at 1 because some servers
        # reject id=0 (it's used as a sentinel "no id" in some specs).
        self._next_id = 1
        # Map of in-flight ``id → _PendingRequest`` so the reader thread
        # can wake the caller that's waiting on it.
        self._pending: Dict[int, _PendingRequest] = {}
        self._lock = threading.Lock()
        # Background reader; daemon=True so it doesn't block process exit.
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Handshake with the server. Required by the MCP spec before any
        # tool calls are allowed.
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
        """Send a JSON-RPC request and synchronously wait for the response.

        The reader thread does the actual matching from id → response.
        We just register a ``_PendingRequest``, write the JSON line out,
        and block on its ``Event``. When the reader receives a response
        with the same id, it stores it on the pending request and signals
        the event.
        """

        # Reserve an id and register the pending request under the lock
        # so the reader thread sees a fully-set-up entry by the time the
        # response can come back.
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
            pending = _PendingRequest(event=threading.Event())
            self._pending[req_id] = pending
        # MCP frames are JSON objects terminated by '\n'. We write the
        # whole thing in one go and flush so the server doesn't have to
        # wait on partial input.
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }) + "\n"
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()
        # Block up to 60 seconds. The event is signalled by ``_read_loop``
        # when a response with our id arrives.
        pending.event.wait(timeout=60)
        # Clean up the pending slot regardless of success — leaks would
        # accumulate over a long-running session.
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
