"""HTTP fetcher. URLs must match an allowlist host suffix."""

from __future__ import annotations

import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Sequence

from ..tool_protocol import Tool, ToolRegistry


def register_http_tools(
    registry: ToolRegistry,
    *,
    allowlist: Optional[Sequence[str]] = None,
    timeout: int = 30,
    max_bytes: int = 2_000_000,
) -> None:
    allowed_hosts = set(allowlist or [])

    def _fetch(args: Dict[str, Any]) -> Dict[str, Any]:
        url = args["url"]
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        if allowed_hosts and not any(host == h or host.endswith("." + h) for h in allowed_hosts):
            raise PermissionError(f"Host {host!r} is not in the HTTP allowlist.")
        req = urllib.request.Request(url, headers={"User-Agent": "chatbot-runtime/1.0"})
        with urllib.request.urlopen(req, timeout=int(args.get("timeout", timeout))) as resp:
            body = resp.read(max_bytes + 1)
            truncated = len(body) > max_bytes
            body = body[:max_bytes]
            text = body.decode("utf-8", errors="replace")
        return {"status": resp.status, "text": text, "truncated": truncated}

    registry.register(Tool(
        name="http_fetch",
        description="Fetch a URL with HTTP GET. Respects allowlist + byte cap.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout": {"type": "integer", "default": timeout},
            },
            "required": ["url"],
        },
        handler=_fetch,
        plugin="builtin.http",
    ))
