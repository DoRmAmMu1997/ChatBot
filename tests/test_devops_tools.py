"""DevOps tool tests — log parsing, clustering, search."""

from __future__ import annotations

from chatbot.runtime.tool_protocol import ToolRegistry
from chatbot.runtime.tools.devops import register_devops_tools


def _registry_with_devops() -> ToolRegistry:
    reg = ToolRegistry()
    register_devops_tools(reg)
    return reg


def test_parse_logs_detects_formats():
    reg = _registry_with_devops()
    parse_logs = reg.get("parse_logs").handler
    payload = (
        "Jun 12 14:22:01 web01 nginx: GET /index.html 200\n"
        '{"level": "ERROR", "msg": "boom", "ts": "2026-05-11T00:00:00Z"}\n'
        "2026-05-11T01:00:01Z INFO checkout-service started\n"
        "this is freeform text\n"
    )
    out = parse_logs({"text": payload})
    assert out["count"] == 4
    formats = {rec["format"] for rec in out["records"]}
    # We should detect at least json + freeform at minimum.
    assert "json" in formats
    assert "freeform" in formats


def test_summarize_clusters_similar_lines():
    reg = _registry_with_devops()
    summarize = reg.get("summarize_incidents").handler
    text = (
        "ERROR pool exhausted size=12\n"
        "ERROR pool exhausted size=13\n"
        "ERROR pool exhausted size=14\n"
        "ERROR disk full bytes=1234\n"
    )
    out = summarize({"text": text, "top_k": 3})
    assert out["total_records"] == 4
    # The "pool exhausted" cluster should be the largest.
    top = out["clusters"][0]
    assert top["count"] == 3
    assert "pool exhausted" in top["representative"]


def test_search_logs_finds_match(tmp_path):
    reg = _registry_with_devops()
    search = reg.get("search_logs").handler
    log_file = tmp_path / "app.log"
    log_file.write_text(
        "before\nERROR boom happened\nafter\n", encoding="utf-8",
    )
    out = search({"path": str(log_file), "pattern": "boom"})
    assert len(out["matches"]) == 1
    assert "boom" in out["matches"][0]["match"]
