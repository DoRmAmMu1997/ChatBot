"""Built-in tools that ship with the runtime."""

from .filesystem import register_filesystem_tools
from .http import register_http_tools
from .notebook import register_notebook_tools
from .shell import register_shell_tools

BUILTIN_REGISTRARS = {
    "builtin.filesystem": register_filesystem_tools,
    "builtin.shell": register_shell_tools,
    "builtin.http": register_http_tools,
    "builtin.notebook": register_notebook_tools,
}

__all__ = [
    "BUILTIN_REGISTRARS",
    "register_filesystem_tools",
    "register_shell_tools",
    "register_http_tools",
    "register_notebook_tools",
]
