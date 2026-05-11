"""Walk plugin directories, validate manifests, register everything they expose."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence

from ..hooks.registry import HookRegistry
from ..skills.loader import load_skill
from ..skills.registry import SkillRegistry
from ..slash_commands.registry import SlashCommandRegistry
from ..tool_protocol import Tool, ToolRegistry
from .manifest import PluginManifest, load_manifest


@dataclass
class PluginInstance:
    """Everything we know about a loaded plugin at runtime."""

    manifest: PluginManifest
    path: Path
    tool_names: List[str] = field(default_factory=list)
    skill_paths: List[Path] = field(default_factory=list)
    slash_commands: List[str] = field(default_factory=list)


def _load_module_from_path(path: Path):
    """Import a single Python file as a module (without polluting sys.path)."""

    if path.suffix != ".py":
        raise ValueError(f"Plugin module must be a .py file: {path}")
    spec = importlib.util.spec_from_file_location(f"chatbot_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_handler(module, dotted: str) -> Callable:
    """``filters.py::sanitize`` → module.sanitize."""

    if "::" not in dotted:
        return getattr(module, dotted)
    file_part, fn = dotted.split("::", 1)
    if file_part:
        # Already loaded above; we just want the function attr.
        pass
    return getattr(module, fn)


def discover_plugins(
    *,
    directories: Sequence[Path | str],
) -> List[Path]:
    """Return paths to all ``plugin.yaml`` files under the given dirs."""

    out: List[Path] = []
    for d in directories:
        root = Path(d).expanduser()
        if not root.exists():
            continue
        for manifest in root.glob("*/plugin.yaml"):
            out.append(manifest)
    return out


def load_plugin(
    manifest_path: Path | str,
    *,
    tool_registry: ToolRegistry,
    hook_registry: HookRegistry,
    skill_registry: SkillRegistry,
    slash_registry: SlashCommandRegistry,
) -> PluginInstance:
    """Load every artefact exposed by a plugin manifest."""

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    plugin_dir = manifest_path.parent
    instance = PluginInstance(manifest=manifest, path=plugin_dir)

    # ---- Tools ----
    for tool_entry in manifest.tools:
        module = _load_module_from_path(plugin_dir / tool_entry.module)
        handler = getattr(module, tool_entry.name)
        tool_registry.register(Tool(
            name=tool_entry.name,
            description=getattr(module, f"{tool_entry.name}__description", manifest.description),
            parameters=tool_entry.schema_ or {"type": "object", "properties": {}},
            handler=handler,
            plugin=manifest.name,
        ))
        instance.tool_names.append(tool_entry.name)

    # ---- Hooks ----
    if manifest.hooks:
        for event, dotted in manifest.hooks.model_dump(exclude_none=True).items():
            file_part, fn_name = dotted.split("::", 1) if "::" in dotted else (None, dotted)
            module_path = plugin_dir / (file_part or "hooks.py")
            module = _load_module_from_path(module_path)
            hook_registry.register(event, getattr(module, fn_name), plugin=manifest.name)

    # ---- Skills ----
    for skill_name in manifest.skills:
        skill_path = plugin_dir / skill_name
        if not skill_path.exists():
            continue
        skill = load_skill(skill_path)
        skill_registry.register(skill)
        instance.skill_paths.append(skill_path)

    # ---- Slash commands ----
    for cmd in manifest.slash_commands:
        file_part, fn_name = cmd.handler.split("::", 1) if "::" in cmd.handler else (None, cmd.handler)
        module_path = plugin_dir / (file_part or "handlers.py")
        module = _load_module_from_path(module_path)
        slash_registry.register(cmd.command, getattr(module, fn_name), plugin=manifest.name)
        instance.slash_commands.append(cmd.command)

    # ---- MCP server (kept for future wire-up) ----
    if manifest.mcp:
        # Spawning the MCP subprocess is the job of mcp_client.py at agent
        # bootstrap time; we just record the intent on the instance.
        pass

    return instance
