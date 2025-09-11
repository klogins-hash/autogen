"""
Base plugin classes and interfaces for the AutoGen plugin system.
"""

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import asyncio
import inspect


class PluginType(Enum):
    """Types of plugins supported by the system."""
    AGENT = "agent"
    TOOL = "tool"
    BEHAVIOR = "behavior"
    MIDDLEWARE = "middleware"
    EXTENSION = "extension"


class PluginCapability(Enum):
    """Capabilities that plugins can provide."""
    MESSAGE_PROCESSING = "message_processing"
    TOOL_EXECUTION = "tool_execution"
    CONTEXT_MODIFICATION = "context_modification"
    ERROR_HANDLING = "error_handling"
    LOGGING = "logging"
    AUTHENTICATION = "authentication"
    CACHING = "caching"
    MONITORING = "monitoring"
    CUSTOM = "custom"


@dataclass
class PluginDependency:
    """Represents a plugin dependency."""
    name: str
    version: Optional[str] = None
    optional: bool = False
    minimum_version: Optional[str] = None
    maximum_version: Optional[str] = None


@dataclass
class PluginConfiguration:
    """Plugin configuration structure."""
    enabled: bool = True
    priority: int = 100
    settings: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[PluginDependency] = field(default_factory=list)


class BasePlugin(abc.ABC):
    """Base class for all AutoGen plugins."""
    
    def __init__(self, config: Optional[PluginConfiguration] = None):
        self.config = config or PluginConfiguration()
        self._initialized = False
        self._active = False
        self._hooks: Dict[str, List[Callable]] = {}
        
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    @property
    @abc.abstractmethod
    def plugin_type(self) -> PluginType:
        """Plugin type."""
        pass
    
    @property
    @abc.abstractmethod
    def capabilities(self) -> Set[PluginCapability]:
        """Plugin capabilities."""
        pass
    
    @property
    def dependencies(self) -> List[PluginDependency]:
        """Plugin dependencies."""
        return self.config.dependencies
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    @property
    def is_active(self) -> bool:
        """Check if plugin is active."""
        return self._active
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            await self._initialize()
            self._initialized = True
            return True
        except Exception as e:
            await self._handle_error(f"Failed to initialize plugin {self.name}: {e}")
            return False
    
    async def activate(self) -> bool:
        """Activate the plugin."""
        if not self._initialized:
            if not await self.initialize():
                return False
        
        try:
            await self._activate()
            self._active = True
            return True
        except Exception as e:
            await self._handle_error(f"Failed to activate plugin {self.name}: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """Deactivate the plugin."""
        try:
            await self._deactivate()
            self._active = False
            return True
        except Exception as e:
            await self._handle_error(f"Failed to deactivate plugin {self.name}: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        try:
            if self._active:
                await self.deactivate()
            await self._shutdown()
            self._initialized = False
            return True
        except Exception as e:
            await self._handle_error(f"Failed to shutdown plugin {self.name}: {e}")
            return False
    
    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook for an event."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def unregister_hook(self, event: str, callback: Callable) -> None:
        """Unregister a hook for an event."""
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)
    
    async def emit_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Emit a hook event and collect results."""
        results = []
        if event in self._hooks:
            for callback in self._hooks[event]:
                try:
                    if inspect.iscoroutinefunction(callback):
                        result = await callback(*args, **kwargs)
                    else:
                        result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    await self._handle_error(f"Hook callback failed: {e}")
        return results
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a plugin setting."""
        return self.config.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a plugin setting."""
        self.config.settings[key] = value
    
    @abc.abstractmethod
    async def _initialize(self) -> None:
        """Plugin-specific initialization logic."""
        pass
    
    async def _activate(self) -> None:
        """Plugin-specific activation logic."""
        pass
    
    async def _deactivate(self) -> None:
        """Plugin-specific deactivation logic."""
        pass
    
    async def _shutdown(self) -> None:
        """Plugin-specific shutdown logic."""
        pass
    
    async def _handle_error(self, error_message: str) -> None:
        """Handle plugin errors."""
        # Default error handling - can be overridden
        print(f"Plugin Error [{self.name}]: {error_message}")
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.plugin_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} v{self.version}>"


class PluginContext:
    """Context object passed to plugins during execution."""
    
    def __init__(
        self,
        agent_name: str = "",
        session_id: str = "",
        message_history: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.agent_name = agent_name
        self.session_id = session_id
        self.message_history = message_history or []
        self.metadata = metadata or {}
        self.shared_data: Dict[str, Any] = {}
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data between plugins."""
        return self.shared_data.get(key, default)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """Set shared data for other plugins."""
        self.shared_data[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get context metadata."""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set context metadata."""
        self.metadata[key] = value


class PluginHook:
    """Decorator for plugin hook methods."""
    
    def __init__(self, event: str, priority: int = 100):
        self.event = event
        self.priority = priority
    
    def __call__(self, func: Callable) -> Callable:
        func._plugin_hook = True
        func._hook_event = self.event
        func._hook_priority = self.priority
        return func


class PluginValidator:
    """Validates plugin implementations."""
    
    @staticmethod
    def validate_plugin(plugin: BasePlugin) -> List[str]:
        """Validate a plugin implementation."""
        errors = []
        
        # Check required properties
        try:
            name = plugin.name
            if not name or not isinstance(name, str):
                errors.append("Plugin name must be a non-empty string")
        except Exception:
            errors.append("Plugin name property is invalid")
        
        try:
            version = plugin.version
            if not version or not isinstance(version, str):
                errors.append("Plugin version must be a non-empty string")
        except Exception:
            errors.append("Plugin version property is invalid")
        
        try:
            plugin_type = plugin.plugin_type
            if not isinstance(plugin_type, PluginType):
                errors.append("Plugin type must be a PluginType enum")
        except Exception:
            errors.append("Plugin type property is invalid")
        
        try:
            capabilities = plugin.capabilities
            if not isinstance(capabilities, set):
                errors.append("Plugin capabilities must be a set")
            elif not all(isinstance(cap, PluginCapability) for cap in capabilities):
                errors.append("All capabilities must be PluginCapability enums")
        except Exception:
            errors.append("Plugin capabilities property is invalid")
        
        # Check required methods
        required_methods = ['_initialize']
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                errors.append(f"Plugin must implement {method_name} method")
            elif not callable(getattr(plugin, method_name)):
                errors.append(f"Plugin {method_name} must be callable")
        
        return errors
    
    @staticmethod
    def validate_dependencies(plugin: BasePlugin, available_plugins: Dict[str, BasePlugin]) -> List[str]:
        """Validate plugin dependencies."""
        errors = []
        
        for dependency in plugin.dependencies:
            if dependency.name not in available_plugins:
                if not dependency.optional:
                    errors.append(f"Required dependency '{dependency.name}' not found")
            else:
                dep_plugin = available_plugins[dependency.name]
                
                # Version checking (simplified)
                if dependency.version and dep_plugin.version != dependency.version:
                    errors.append(f"Dependency '{dependency.name}' version mismatch")
        
        return errors


# Utility functions for plugin development
def plugin_hook(event: str, priority: int = 100):
    """Decorator for marking plugin hook methods."""
    return PluginHook(event, priority)


def get_plugin_hooks(plugin: BasePlugin) -> Dict[str, List[Callable]]:
    """Extract hook methods from a plugin."""
    hooks = {}
    
    for attr_name in dir(plugin):
        attr = getattr(plugin, attr_name)
        if callable(attr) and hasattr(attr, '_plugin_hook'):
            event = attr._hook_event
            if event not in hooks:
                hooks[event] = []
            hooks[event].append((attr._hook_priority, attr))
    
    # Sort by priority
    for event in hooks:
        hooks[event].sort(key=lambda x: x[0])
        hooks[event] = [hook[1] for hook in hooks[event]]
    
    return hooks
