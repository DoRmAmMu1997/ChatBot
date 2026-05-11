# Plugin / skill / hook / slash-command system

Forge (and Aurora, if you flip `tools_enabled: true` in its runtime
config) ships with an agent layer modeled after Claude Code. This
document is the field manual for that layer.

## The mental model

```
   user input
       │
       ▼
   ┌──────────────────────────┐
   │ slash-command resolution │ ── /help, /skills, /tools, custom
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ skill auto-selection     │ ── match triggers; prepend to system prompt
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ on_message hooks         │
   └────────────┬─────────────┘
                ▼
        model.generate()
                │
                ▼
   ┌──────────────────────────┐
   │ parse <|tool_call|> …    │
   └────┬─────────────────────┘
        │                     no tool calls
        │ ─────────────────────────────────▶ on_stop hooks → return reply
        │
        ▼ pre_tool hooks   →  may skip / rewrite arguments
   ┌──────────────────────────┐
   │ ToolRegistry dispatch    │
   └────────────┬─────────────┘
                ▼ post_tool hooks  →  may rewrite the result
        append <|tool_result|>
                │
                └──── loop back to model.generate()
```

The loop is bounded by `tool_max_iterations` in the runtime config (default
24 for Aurora, 64 for Forge). When the budget is exhausted we surface
`on_stop` with `reason="max_iterations"`.

## Plugin layout

A plugin is a directory containing a `plugin.yaml`. Example:

```
my_plugin/
├── plugin.yaml
├── tools.py
├── hooks.py
├── handlers.py
└── skills/
    └── search.md
```

`plugin.yaml`:

```yaml
name: my_plugin
version: 0.1.0
description: Adds a custom search tool, a redaction hook, and /find slash command.
tools:
  - module: tools.py
    name: search
    schema:
      type: object
      properties:
        query: { type: string }
        limit: { type: integer, default: 10 }
      required: [query]
hooks:
  pre_tool: hooks.py::redact_secrets
slash_commands:
  - command: /find
    handler: handlers.py::find
skills:
  - skills/search.md
mcp:                 # optional: forward to an MCP server
  command: ["python", "-m", "my_mcp_server"]
  transport: stdio
```

## Tools

A tool is a Python callable registered in `plugin.yaml::tools`. Each
takes a single `dict` of arguments and returns any JSON-serializable
value. The schema is JSON-Schema; the model sees the schema in its system
prompt and is expected to emit a matching `<|tool_call|>` block.

The runtime ships four built-in tool families (enable per runtime config):
* `builtin.filesystem` — `read_file`, `write_file`, `glob`, `grep`
* `builtin.shell` — `shell` (subprocess + allowlist + timeout)
* `builtin.http` — `http_fetch`
* `builtin.notebook` — `python` (stateful exec)

## Skills

Skills are markdown files with YAML frontmatter:

```markdown
---
name: code_review
description: Structured PR review.
triggers:
  - "code review"
  - "review the PR"
---

When the user asks for a code review, follow these steps … (markdown body)
```

If `triggers` match the latest user message, the body is prepended to the
system prompt for that turn. To always-on a skill regardless of triggers,
register a slash command that activates it.

## Hooks

Four lifecycle events. Each hook is a plain function `(payload: dict) -> dict | None`.

| Event | Payload | Use case |
|---|---|---|
| `on_message` | `{"message": AgentMessage}` | Audit / redact incoming text. |
| `pre_tool` | `{"tool_name": str, "arguments": dict, "skip": bool}` | Block / rewrite tool calls. Set `payload["skip"]=True` to cancel. |
| `post_tool` | `{"tool_name": str, "arguments": dict, "result": Any}` | Sanitize results before the model sees them. |
| `on_stop` | `{"history": list, "reason": str}` | Telemetry. |

## Slash commands

Resolved before the model sees the message. Return a string to replace
the user's input, or `None` to fall through. Built-in commands:

| Command | What it does |
|---|---|
| `/help` | Lists tools + skills + commands. |
| `/skills` | Lists loaded skills. |
| `/tools` | Lists registered tools. |

## MCP servers

If a plugin's `plugin.yaml` declares an `mcp:` block, the runtime spawns
that subprocess and uses the line-framed JSON-RPC protocol to call its
tools alongside the local registry. See
`src/chatbot/runtime/plugins/mcp_client.py` for the implementation.

## Subagents

Sub-agents run an isolated copy of the loop with a restricted toolset and
no shared history. Implementation lives in `src/chatbot/runtime/subagents.py`.
They share the same model instance — we don't load the weights twice.

## Testing your plugin

Drop your plugin into `~/.chatbot/plugins/` (or anywhere on
`runtime.extra_plugin_paths`) and start the agent:

```powershell
python scripts/agent.py `
    --model tiny `
    --checkpoint outputs/sft/latest `
    --tokenizer checkpoints/tiny-tok.json `
    --runtime forge-coder
```

Type `/help` to see your tools and slash commands. The unit tests in
`tests/test_plugin_loader.py` show how to load a plugin in code.
