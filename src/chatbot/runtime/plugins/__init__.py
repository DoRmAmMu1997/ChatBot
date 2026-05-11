"""Plugin discovery and loading."""

from .loader import PluginInstance, discover_plugins, load_plugin
from .manifest import PluginManifest, load_manifest

__all__ = [
    "PluginInstance",
    "PluginManifest",
    "discover_plugins",
    "load_manifest",
    "load_plugin",
]
