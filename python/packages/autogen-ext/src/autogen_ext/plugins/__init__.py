"""
Plugin architecture for extensible AutoGen agents.

This module provides a flexible plugin system that allows:
- Dynamic loading of agent capabilities
- Custom tool registration
- Behavior modification through plugins
- Runtime plugin management
"""

from ._plugin_manager import PluginManager, PluginInfo, PluginStatus
from ._base_plugin import BasePlugin, PluginType, PluginCapability
from ._agent_plugin import AgentPlugin, ToolPlugin, BehaviorPlugin
from ._plugin_registry import PluginRegistry, PluginMetadata
from ._plugin_loader import PluginLoader, PluginLoadError

__all__ = [
    "PluginManager",
    "PluginInfo", 
    "PluginStatus",
    "BasePlugin",
    "PluginType",
    "PluginCapability",
    "AgentPlugin",
    "ToolPlugin", 
    "BehaviorPlugin",
    "PluginRegistry",
    "PluginMetadata",
    "PluginLoader",
    "PluginLoadError"
]
