"""Plugin manifest (``plugin.yaml``) schema and parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ToolEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module: str                          # Python file relative to plugin dir
    name: str                            # Tool name registered on the bus
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")


class HookEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pre_tool: Optional[str] = None
    post_tool: Optional[str] = None
    on_message: Optional[str] = None
    on_stop: Optional[str] = None


class SlashEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str
    handler: str


class MCPEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: List[str]
    transport: str = "stdio"


class PluginManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str = "0.0.0"
    description: str = ""
    tools: List[ToolEntry] = Field(default_factory=list)
    hooks: Optional[HookEntry] = None
    skills: List[str] = Field(default_factory=list)
    slash_commands: List[SlashEntry] = Field(default_factory=list)
    mcp: Optional[MCPEntry] = None


def load_manifest(path: str | Path) -> PluginManifest:
    """Parse ``plugin.yaml`` into a typed :class:`PluginManifest`."""

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PluginManifest.model_validate(raw)
