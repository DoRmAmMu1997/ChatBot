# Example plugins

Two reference plugins live here. Copy either folder into
`~/.chatbot/plugins/` (or anywhere on the runtime's `extra_plugin_paths`)
and Forge will load it on startup.

## `code_review/`

Demonstrates a **skill** — a markdown file with YAML frontmatter that the
runtime auto-prefixes into the system prompt whenever the user message
contains a matching trigger phrase. No code, just guidance.

## `filesystem_mcp/`

Demonstrates a full plugin manifest with:
* a Python module that exposes a tool callable,
* a hook that runs before every tool call (logs the call),
* a slash command (`/wd`) that prints the current working directory.

It also includes a stub MCP server entry point so you can see how an
external MCP integration plugs in, even though Forge ships built-in tools
that cover the same surface.
