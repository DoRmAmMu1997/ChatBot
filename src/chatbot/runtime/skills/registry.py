"""In-memory directory of loaded :class:`Skill` objects."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .loader import Skill


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all(self) -> List[Skill]:
        return list(self._skills.values())

    def match(self, query: str, *, max_active: int = 4) -> List[Skill]:
        """Find skills whose trigger substrings appear in ``query``.

        Empty triggers means the skill is never auto-selected — it must be
        invoked explicitly via slash command.
        """

        query_lower = query.lower()
        matches: List[Skill] = []
        for skill in self._skills.values():
            if not skill.triggers:
                continue
            if any(trigger.lower() in query_lower for trigger in skill.triggers):
                matches.append(skill)
                if len(matches) >= max_active:
                    break
        return matches
