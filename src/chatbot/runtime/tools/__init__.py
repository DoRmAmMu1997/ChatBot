"""Built-in tools that ship with the runtime.

Each entry in :data:`BUILTIN_REGISTRARS` is a function that takes a
:class:`ToolRegistry` and adds its tools. The runtime config's
``enabled_plugins`` list names which of these to wire on startup.
"""

from .devops import register_devops_tools
from .document import register_document_tools
from .filesystem import register_filesystem_tools
from .http import register_http_tools
from .notebook import register_notebook_tools
from .shell import register_shell_tools

BUILTIN_REGISTRARS = {
    "builtin.filesystem": register_filesystem_tools,
    "builtin.shell": register_shell_tools,
    "builtin.http": register_http_tools,
    "builtin.notebook": register_notebook_tools,
    "builtin.devops": register_devops_tools,
    "builtin.document": register_document_tools,
}

__all__ = [
    "BUILTIN_REGISTRARS",
    "register_filesystem_tools",
    "register_shell_tools",
    "register_http_tools",
    "register_notebook_tools",
    "register_devops_tools",
    "register_document_tools",
]
