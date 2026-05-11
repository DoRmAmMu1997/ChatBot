"""Synthetic log-triage data — pairs of "noisy logs" + "the correct fix".

A real production log corpus is gold for training DevOps-style models, but
they're rarely public. This module fabricates plausible incidents
(stack traces, 5xx spikes, OOM kills, slow queries) and pairs each with a
hand-written analysis + remediation. Useful as extra signal during the
DevOps SFT stage; not a substitute for real incidents.
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass
from typing import Iterator, List


@dataclass(frozen=True)
class LogTriageExample:
    issue_kind: str
    log_payload: str
    analysis: str           # what the model should say it sees.
    remediation: str        # concrete fix.


_TEMPLATES = [
    {
        "kind": "5xx_spike",
        "log_template": (
            "2026-05-11T14:{minute:02d}:{second:02d}Z ERROR app=checkout "
            "trace_id={tid} 500 POST /v1/checkout cart_size={cart} "
            "msg='database connection pool exhausted'"
        ),
        "lines": 6,
        "analysis": (
            "Every error is the same: 'database connection pool exhausted' on POST "
            "/v1/checkout. The checkout service is opening connections faster than "
            "it can return them; the issue starts when cart size grows."
        ),
        "remediation": (
            "1. Raise the pool size from 10 to 40 (matches the request rate during peak).\n"
            "2. Audit the checkout handler for missing `with` / context-manager exits.\n"
            "3. Add a circuit-breaker that fast-fails after a sustained 95%+ pool usage so "
            "downstream services don't pile up."
        ),
    },
    {
        "kind": "oom",
        "log_template": (
            "May 11 14:{minute:02d}:{second:02d} kernel: [PID {pid}] Out of memory: "
            "Killed process {pid} (api-worker) total-vm:8197312kB anon-rss:7892108kB"
        ),
        "lines": 4,
        "analysis": (
            "OOM-kill repeating on `api-worker`. Each killed process is sitting at "
            "~7.5 GB anon RSS just before death — a clear leak rather than a momentary "
            "burst."
        ),
        "remediation": (
            "1. Cap the container memory at 6 GB so workers OOM faster and you can repro.\n"
            "2. Take a heap snapshot at the 5-GB mark (e.g. via `tracemalloc.snapshot`).\n"
            "3. Most-common suspect: the response cache lacks a max-size; switch the "
            "in-process LRU to bounded (256 MB) or move it to Redis."
        ),
    },
    {
        "kind": "stack_trace",
        "log_template": (
            "Traceback (most recent call last):\n"
            "  File \"/app/orders.py\", line {line}, in process_order\n"
            "    return ship(order['shipping_address']['country'])\n"
            "KeyError: 'shipping_address'"
        ),
        "lines": 1,
        "analysis": (
            "A KeyError on `'shipping_address'` is firing for any order created via "
            "the new /v2/orders endpoint — the field was renamed to `delivery_address` "
            "in that path but the consumer hasn't been updated."
        ),
        "remediation": (
            "1. Patch `process_order` to read either field:\n"
            "   `addr = order.get('shipping_address') or order.get('delivery_address')`.\n"
            "2. Add a regression test that covers /v2/orders payloads.\n"
            "3. File a ticket to align the field name across the schema."
        ),
    },
    {
        "kind": "slow_query",
        "log_template": (
            "2026-05-11T14:{minute:02d}:{second:02d}Z postgres duration={duration}ms "
            "statement='SELECT * FROM events WHERE customer_id={cid} ORDER BY created_at DESC'"
        ),
        "lines": 5,
        "analysis": (
            "All slow queries (>2 s) are the same `SELECT * FROM events WHERE "
            "customer_id=? ORDER BY created_at DESC`. Without a `(customer_id, "
            "created_at)` index, Postgres is scanning the entire `events` table per "
            "customer."
        ),
        "remediation": (
            "1. `CREATE INDEX CONCURRENTLY idx_events_customer_created "
            "ON events(customer_id, created_at DESC);`.\n"
            "2. Add a LIMIT to the application query if callers only need the last N.\n"
            "3. Plan a periodic `VACUUM ANALYZE` cadence — table is hot enough that "
            "autovacuum may be falling behind."
        ),
    },
]


def _render_lines(template: dict, rng: random.Random) -> str:
    """Build a single log payload from a template by interpolating random fields."""

    lines: List[str] = []
    for _ in range(int(template["lines"])):
        line = template["log_template"].format(
            minute=rng.randint(0, 59),
            second=rng.randint(0, 59),
            tid=hex(rng.randint(0, 2**64))[-8:],
            pid=rng.randint(1000, 9999),
            cart=rng.randint(1, 50),
            line=rng.randint(10, 500),
            duration=rng.randint(2000, 9000),
            cid=rng.randint(1, 1_000_000),
        )
        lines.append(line)
    return "\n".join(lines)


def iter_log_triage_examples(*, seed: int = 0, num: int = 1000) -> Iterator[LogTriageExample]:
    """Yield up to ``num`` log-triage examples sampled from the templates."""

    rng = random.Random(seed)
    for _ in range(num):
        template = rng.choice(_TEMPLATES)
        yield LogTriageExample(
            issue_kind=template["kind"],
            log_payload=_render_lines(template, rng),
            analysis=textwrap.fill(template["analysis"], width=88),
            remediation=textwrap.dedent(template["remediation"]),
        )


def example_to_chat_messages(example: LogTriageExample) -> List[dict]:
    """Render a triage example as a chat-message list (system / user / assistant)."""

    return [
        {"role": "system", "content":
            "You are an experienced SRE. Read the logs, state what's happening in one "
            "paragraph, then propose a concrete remediation."},
        {"role": "user", "content":
            "Here is a batch of recent logs:\n\n```\n" + example.log_payload + "\n```\n\n"
            "What's going on, and what should we do about it?"},
        {"role": "assistant", "content":
            example.analysis + "\n\nRemediation:\n" + example.remediation},
    ]
