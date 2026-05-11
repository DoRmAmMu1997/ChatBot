"""Parse a markdown file with YAML frontmatter into a :class:`Skill`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass
class Skill:
    """A reusable chunk of guidance the agent can inject into the system prompt.

    Fields are intentionally Claude-Code-aligned: ``name``, ``description``,
    ``triggers`` (substring matches), ``body`` (the markdown body), and
    ``path`` (source file for debugging).
    """

    name: str
    description: str
    body: str
    path: Optional[Path] = None
    triggers: List[str] = field(default_factory=list)


def load_skill(path: Path | str) -> Skill:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter — treat entire file as body, derive name from filename.
        return Skill(name=path.stem, description="", body=text, path=path)
    fm = yaml.safe_load(match.group(1)) or {}
    body = text[match.end() :]
    return Skill(
        name=str(fm.get("name", path.stem)),
        description=str(fm.get("description", "")),
        triggers=list(fm.get("triggers", [])),
        body=body,
        path=path,
    )


def load_skills_from_directory(directory: Path | str) -> List[Skill]:
    out: List[Skill] = []
    for path in Path(directory).glob("*.md"):
        try:
            out.append(load_skill(path))
        except Exception:
            continue
    return out
