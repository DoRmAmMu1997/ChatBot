"""DevOps / SRE built-in tools — log parsing, tail, search, incident summary, metrics.

Forge uses these heavily when the user pastes raw logs or asks about
an incident. They're pure-Python, depend only on stdlib + regex, and live
behind the ``builtin.devops`` plugin name in the runtime's enabled list.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..tool_protocol import Tool, ToolRegistry


# ---- Log parsers --------------------------------------------------------

_SYSLOG_RE = re.compile(
    r"^(?P<ts>\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<source>\S+?):\s*(?P<msg>.*)$"
)
_K8S_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+"
    r"(?P<level>\w+)\s+(?P<source>\S+)\s+(?P<msg>.*)$"
)
_APACHE_RE = re.compile(
    r'^(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<path>\S+)[^"]*"\s+'
    r'(?P<status>\d+)\s+(?P<bytes>\S+)'
)


def _try_parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """Return a parsed dict if the line is a single JSON object, else None.

    We bail early if the line doesn't look like ``{ ... }`` so we don't pay
    the JSON parser's price on plain-text lines.
    """

    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _parse_one(line: str) -> Dict[str, Any]:
    """Try each known format; if none match, fall back to a free-text record.

    Returns a uniform schema (``timestamp``, ``level``, ``source``,
    ``message``, ``raw``, ``format``) regardless of what the original line
    looked like, so downstream code (clustering, search) doesn't need to
    care which format the user pasted.
    """

    # JSON-lines are common in modern services — try those first since they
    # carry the most structure.
    j = _try_parse_json_line(line)
    if j is not None:
        return {
            "timestamp": j.get("ts") or j.get("time") or j.get("timestamp"),
            "level": j.get("level") or j.get("severity"),
            "source": j.get("source") or j.get("logger"),
            "message": j.get("msg") or j.get("message") or json.dumps(j),
            "raw": line,
            "format": "json",
        }
    # Try each regex in order; first match wins. Order matters: K8s logs
    # are timestamp-prefixed and similar to syslog, but the regex is
    # stricter, so we try it first.
    for fmt, rx in (("k8s", _K8S_RE), ("syslog", _SYSLOG_RE), ("apache", _APACHE_RE)):
        m = rx.match(line)
        if m:
            d = m.groupdict()
            return {
                "timestamp": d.get("ts"),
                "level": d.get("level"),
                "source": d.get("source") or d.get("host"),
                "message": d.get("msg") or d.get("path"),
                "raw": line,
                "format": fmt,
            }
    # Nothing matched — treat as free-form. We still emit the uniform
    # schema with ``message`` set to the whole line.
    return {
        "timestamp": None, "level": None, "source": None,
        "message": line, "raw": line, "format": "freeform",
    }


def _normalize_for_cluster(msg: str) -> str:
    """Replace numeric / id-like substrings so similar messages collapse.

    Two error lines that differ only in a trace id or timestamp should
    cluster together. We strip the four common variable patterns:

    * IPv4 addresses → ``<IP>``
    * 0xHEX literals → ``<HEX>``
    * long lowercase hex strings (trace ids, hashes) → ``<HASH>``
    * plain integers → ``<N>``

    Lowercasing makes "ERROR" and "Error" collide. The result isn't
    meant to be read by humans — it's just a key for the similarity
    counter in ``summarize_incidents``.
    """

    msg = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", msg)
    msg = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", msg)
    msg = re.sub(r"\b[0-9a-f]{16,}\b", "<HASH>", msg)
    msg = re.sub(r"\b\d+\b", "<N>", msg)
    return msg.lower()


def register_devops_tools(registry: ToolRegistry) -> None:
    def parse_logs(args: Dict[str, Any]) -> Dict[str, Any]:
        text: str = args.get("text", "")
        path: Optional[str] = args.get("path")
        if path:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        records = [_parse_one(line) for line in text.splitlines() if line.strip()]
        return {"records": records, "count": len(records)}

    def tail_log_file(args: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(args["path"])
        n = int(args.get("n", 200))
        level_filter = args.get("level")
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if level_filter:
            lines = [ln for ln in lines if level_filter.upper() in ln.upper()]
        return {"lines": lines[-n:]}

    def search_logs(args: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(args.get("path", "."))
        pattern = re.compile(args["pattern"])
        before = int(args.get("context_before", 1))
        after = int(args.get("context_after", 1))
        limit = int(args.get("limit", 200))
        out: List[Dict[str, Any]] = []
        candidates: Sequence[Path] = (
            [root] if root.is_file() else list(root.rglob("*.log")) + list(root.rglob("*.txt"))
        )
        for p in candidates:
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines):
                if pattern.search(line):
                    ctx_start = max(0, i - before)
                    ctx_end = min(len(lines), i + after + 1)
                    out.append({
                        "path": str(p),
                        "line_no": i + 1,
                        "match": line,
                        "context": lines[ctx_start:ctx_end],
                    })
                    if len(out) >= limit:
                        return {"matches": out}
        return {"matches": out}

    def summarize_incidents(args: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster log messages by similarity; return top-K representatives.

        The simplest possible clustering: strip variables, count exact
        matches. Good enough to find the "what's actually happening" view
        of a noisy log payload.
        """

        records = args.get("records")
        if records is None:
            # Allow passing raw text too.
            parsed = parse_logs({"text": args.get("text", "")})
            records = parsed["records"]
        k = int(args.get("top_k", 5))
        counter: Counter = Counter()
        examples: Dict[str, str] = {}
        for record in records:
            msg = (record.get("message") or record.get("raw") or "").strip()
            if not msg:
                continue
            key = _normalize_for_cluster(msg)
            counter[key] += 1
            examples.setdefault(key, msg)
        return {
            "clusters": [
                {"count": cnt, "representative": examples[key]}
                for key, cnt in counter.most_common(k)
            ],
            "total_records": sum(counter.values()),
        }

    def query_metric(args: Dict[str, Any]) -> Dict[str, Any]:
        """PromQL-style metrics query.

        Default backend returns a deterministic stub — useful for tests and
        demos. Set the ``CHATBOT_METRICS_URL`` env var to a real Prometheus
        ``/api/v1/query_range`` endpoint and we'll forward the call.
        """

        url = os.environ.get("CHATBOT_METRICS_URL")
        if not url:
            return {
                "status": "stub",
                "warning": "No CHATBOT_METRICS_URL configured — returning a stub series.",
                "query": args.get("query"),
                "result": [
                    {"metric": {"__name__": str(args.get("query", "stub"))},
                     "values": [[i, float(i * 0.1)] for i in range(10)]},
                ],
            }
        params = urllib.parse.urlencode({
            "query": args["query"],
            "start": args.get("start", ""),
            "end": args.get("end", ""),
            "step": args.get("step", "15s"),
        })
        req = urllib.request.Request(f"{url}?{params}",
                                      headers={"User-Agent": "chatbot-runtime/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    registry.register(Tool(
        name="parse_logs",
        description="Parse a log payload (text or path) into structured records.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "path": {"type": "string"},
            },
        },
        handler=parse_logs,
        plugin="builtin.devops",
    ))
    registry.register(Tool(
        name="tail_log_file",
        description="Return the last N lines of a log file, optionally filtered by level.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "n": {"type": "integer", "default": 200},
                "level": {"type": "string", "description": "Substring filter, e.g. 'ERROR'."},
            },
            "required": ["path"],
        },
        handler=tail_log_file,
        plugin="builtin.devops",
    ))
    registry.register(Tool(
        name="search_logs",
        description="Regex search across log files in a tree, with context windows.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string"},
                "context_before": {"type": "integer", "default": 1},
                "context_after": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 200},
            },
            "required": ["pattern"],
        },
        handler=search_logs,
        plugin="builtin.devops",
    ))
    registry.register(Tool(
        name="summarize_incidents",
        description="Cluster log messages by similarity and return the top-K representatives.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "records": {"type": "array", "items": {"type": "object"}},
                "top_k": {"type": "integer", "default": 5},
            },
        },
        handler=summarize_incidents,
        plugin="builtin.devops",
    ))
    registry.register(Tool(
        name="query_metric",
        description="PromQL-style query against a Prometheus-compatible backend.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "step": {"type": "string", "default": "15s"},
            },
            "required": ["query"],
        },
        handler=query_metric,
        plugin="builtin.devops",
    ))
