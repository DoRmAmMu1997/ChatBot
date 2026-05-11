"""Slash commands — shortcut handlers triggered by ``/<name>`` user input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# A handler takes the rest of the user's line (without the slash + name) and
# returns a string that will replace the user message. Return ``None`` to
# fall through to the model unchanged.
SlashHandler = Callable[[str], Optional[str]]


@dataclass
class _Entry:
    handler: SlashHandler
    plugin: Optional[str] = None


class SlashCommandRegistry:
    def __init__(self):
        self._commands: Dict[str, _Entry] = {}

    def register(self, command: str, handler: SlashHandler, *, plugin: Optional[str] = None) -> None:
        # Allow both ``/lint`` and ``lint`` as keys — internally normalize.
        key = command.lstrip("/").strip()
        self._commands[key] = _Entry(handler=handler, plugin=plugin)

    def names(self) -> List[str]:
        return sorted(self._commands)

    def resolve(self, line: str) -> Optional[str]:
        """If ``line`` starts with a registered ``/cmd``, dispatch and return its result."""

        if not line.startswith("/"):
            return None
        head, _, rest = line[1:].partition(" ")
        entry = self._commands.get(head)
        if entry is None:
            return None
        return entry.handler(rest.strip())
