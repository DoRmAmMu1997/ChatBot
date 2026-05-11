"""Plugin loader tests using the shipped example plugin."""

from __future__ import annotations

from pathlib import Path

from chatbot.runtime.hooks.registry import HookRegistry
from chatbot.runtime.plugins.loader import discover_plugins, load_plugin
from chatbot.runtime.skills.registry import SkillRegistry
from chatbot.runtime.slash_commands.registry import SlashCommandRegistry
from chatbot.runtime.tool_protocol import ToolRegistry


def test_discover_finds_example_plugins():
    repo_root = Path(__file__).resolve().parent.parent
    found = discover_plugins(directories=[repo_root / "plugins_examples"])
    names = {p.parent.name for p in found}
    assert "filesystem_mcp" in names
    assert "code_review" in names


def test_load_filesystem_mcp_example():
    repo_root = Path(__file__).resolve().parent.parent
    manifest = repo_root / "plugins_examples" / "filesystem_mcp" / "plugin.yaml"
    tools = ToolRegistry()
    hooks = HookRegistry()
    skills = SkillRegistry()
    slash = SlashCommandRegistry()
    instance = load_plugin(
        manifest,
        tool_registry=tools, hook_registry=hooks,
        skill_registry=skills, slash_registry=slash,
    )
    assert "word_count" in tools
    assert "/wd" in slash.names() or "wd" in slash.names()
    assert hooks.hooks_for("pre_tool")
    assert instance.manifest.name == "filesystem_mcp_example"


def test_load_code_review_example():
    repo_root = Path(__file__).resolve().parent.parent
    manifest = repo_root / "plugins_examples" / "code_review" / "plugin.yaml"
    tools = ToolRegistry()
    hooks = HookRegistry()
    skills = SkillRegistry()
    slash = SlashCommandRegistry()
    load_plugin(
        manifest,
        tool_registry=tools, hook_registry=hooks,
        skill_registry=skills, slash_registry=slash,
    )
    assert skills.get("code_review") is not None
