"""
Plugin manager for AutoGen plugin system.
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union
import importlib
import sys

from ._base_plugin import BasePlugin, PluginType, PluginCapability, PluginValidator, PluginContext
from ._plugin_loader import PluginLoader, PluginLoadError


class PluginStatus(Enum):
    """Plugin status states."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    description: str
    plugin_type: PluginType
    capabilities: Set[PluginCapability]
    status: PluginStatus
    plugin_instance: Optional[BasePlugin] = None
    error_message: Optional[str] = None
    load_time: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)


class PluginManager:
    """Manages the lifecycle of AutoGen plugins."""
    
    def __init__(
        self,
        plugin_directories: Optional[List[str]] = None,
        config_file: Optional[str] = None,
        auto_load: bool = True
    ):
        self.plugin_directories = plugin_directories or ["./plugins"]
        self.config_file = config_file
        self.auto_load = auto_load
        
        # Plugin storage
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_loader = PluginLoader()
        
        # Plugin organization
        self.plugins_by_type: Dict[PluginType, List[str]] = {pt: [] for pt in PluginType}
        self.plugins_by_capability: Dict[PluginCapability, List[str]] = {pc: [] for pc in PluginCapability}
        
        # Event hooks
        self.event_hooks: Dict[str, List[BasePlugin]] = {}
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        if config_file:
            self._load_config()
        
        if auto_load:
            asyncio.create_task(self._auto_load_plugins())
    
    async def load_plugin(
        self,
        plugin_path: str,
        plugin_name: Optional[str] = None
    ) -> bool:
        """Load a plugin from a file or module path."""
        
        try:
            plugin_class = await self.plugin_loader.load_plugin_class(plugin_path)
            
            if not plugin_class:
                return False
            
            # Create plugin instance
            plugin_config = self.config.get("plugins", {}).get(plugin_name or plugin_class.__name__, {})
            plugin_instance = plugin_class(plugin_config)
            
            # Validate plugin
            validation_errors = PluginValidator.validate_plugin(plugin_instance)
            if validation_errors:
                await self._handle_plugin_error(
                    plugin_instance.name,
                    f"Plugin validation failed: {'; '.join(validation_errors)}"
                )
                return False
            
            # Check dependencies
            dependency_errors = PluginValidator.validate_dependencies(
                plugin_instance,
                {name: info.plugin_instance for name, info in self.plugins.items() 
                 if info.plugin_instance}
            )
            if dependency_errors:
                await self._handle_plugin_error(
                    plugin_instance.name,
                    f"Dependency validation failed: {'; '.join(dependency_errors)}"
                )
                return False
            
            # Create plugin info
            plugin_info = PluginInfo(
                name=plugin_instance.name,
                version=plugin_instance.version,
                description=plugin_instance.description,
                plugin_type=plugin_instance.plugin_type,
                capabilities=plugin_instance.capabilities,
                status=PluginStatus.LOADED,
                plugin_instance=plugin_instance,
                load_time=asyncio.get_event_loop().time(),
                dependencies=[dep.name for dep in plugin_instance.dependencies]
            )
            
            # Store plugin
            self.plugins[plugin_instance.name] = plugin_info
            self.plugins_by_type[plugin_instance.plugin_type].append(plugin_instance.name)
            
            for capability in plugin_instance.capabilities:
                self.plugins_by_capability[capability].append(plugin_instance.name)
            
            return True
            
        except Exception as e:
            await self._handle_plugin_error(plugin_name or "unknown", f"Failed to load plugin: {e}")
            return False
    
    async def initialize_plugin(self, plugin_name: str) -> bool:
        """Initialize a loaded plugin."""
        
        plugin_info = self.plugins.get(plugin_name)
        if not plugin_info or not plugin_info.plugin_instance:
            return False
        
        if plugin_info.status != PluginStatus.LOADED:
            return False
        
        try:
            success = await plugin_info.plugin_instance.initialize()
            if success:
                plugin_info.status = PluginStatus.INITIALIZED
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Initialization failed"
                return False
                
        except Exception as e:
            await self._handle_plugin_error(plugin_name, f"Initialization failed: {e}")
            return False
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """Activate an initialized plugin."""
        
        plugin_info = self.plugins.get(plugin_name)
        if not plugin_info or not plugin_info.plugin_instance:
            return False
        
        if plugin_info.status not in [PluginStatus.INITIALIZED, PluginStatus.LOADED]:
            return False
        
        try:
            success = await plugin_info.plugin_instance.activate()
            if success:
                plugin_info.status = PluginStatus.ACTIVE
                await self._register_plugin_hooks(plugin_info.plugin_instance)
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Activation failed"
                return False
                
        except Exception as e:
            await self._handle_plugin_error(plugin_name, f"Activation failed: {e}")
            return False
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate an active plugin."""
        
        plugin_info = self.plugins.get(plugin_name)
        if not plugin_info or not plugin_info.plugin_instance:
            return False
        
        if plugin_info.status != PluginStatus.ACTIVE:
            return False
        
        try:
            success = await plugin_info.plugin_instance.deactivate()
            if success:
                plugin_info.status = PluginStatus.INITIALIZED
                await self._unregister_plugin_hooks(plugin_info.plugin_instance)
                return True
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Deactivation failed"
                return False
                
        except Exception as e:
            await self._handle_plugin_error(plugin_name, f"Deactivation failed: {e}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin completely."""
        
        plugin_info = self.plugins.get(plugin_name)
        if not plugin_info:
            return False
        
        try:
            # Deactivate if active
            if plugin_info.status == PluginStatus.ACTIVE:
                await self.deactivate_plugin(plugin_name)
            
            # Shutdown if initialized
            if plugin_info.plugin_instance and plugin_info.status in [PluginStatus.INITIALIZED, PluginStatus.ERROR]:
                await plugin_info.plugin_instance.shutdown()
            
            # Remove from indices
            if plugin_info.plugin_type in self.plugins_by_type:
                if plugin_name in self.plugins_by_type[plugin_info.plugin_type]:
                    self.plugins_by_type[plugin_info.plugin_type].remove(plugin_name)
            
            for capability in plugin_info.capabilities:
                if plugin_name in self.plugins_by_capability[capability]:
                    self.plugins_by_capability[capability].remove(plugin_name)
            
            # Remove plugin
            del self.plugins[plugin_name]
            return True
            
        except Exception as e:
            await self._handle_plugin_error(plugin_name, f"Unload failed: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str, plugin_path: Optional[str] = None) -> bool:
        """Reload a plugin."""
        
        # Store current path if not provided
        if not plugin_path and plugin_name in self.plugins:
            # In a real implementation, you'd store the original path
            plugin_path = f"./plugins/{plugin_name}.py"
        
        # Unload current plugin
        await self.unload_plugin(plugin_name)
        
        # Load new version
        return await self.load_plugin(plugin_path, plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin instance by name."""
        
        plugin_info = self.plugins.get(plugin_name)
        return plugin_info.plugin_instance if plugin_info else None
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all active plugins of a specific type."""
        
        plugins = []
        for plugin_name in self.plugins_by_type.get(plugin_type, []):
            plugin_info = self.plugins.get(plugin_name)
            if plugin_info and plugin_info.status == PluginStatus.ACTIVE and plugin_info.plugin_instance:
                plugins.append(plugin_info.plugin_instance)
        
        return plugins
    
    def get_plugins_by_capability(self, capability: PluginCapability) -> List[BasePlugin]:
        """Get all active plugins with a specific capability."""
        
        plugins = []
        for plugin_name in self.plugins_by_capability.get(capability, []):
            plugin_info = self.plugins.get(plugin_name)
            if plugin_info and plugin_info.status == PluginStatus.ACTIVE and plugin_info.plugin_instance:
                plugins.append(plugin_info.plugin_instance)
        
        return plugins
    
    async def execute_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Execute all registered hooks for an event."""
        
        results = []
        
        if event in self.event_hooks:
            for plugin in self.event_hooks[event]:
                try:
                    plugin_results = await plugin.emit_hook(event, *args, **kwargs)
                    results.extend(plugin_results)
                except Exception as e:
                    await self._handle_plugin_error(plugin.name, f"Hook execution failed: {e}")
        
        return results
    
    async def load_plugins_from_directory(self, directory: str) -> int:
        """Load all plugins from a directory."""
        
        directory_path = Path(directory)
        if not directory_path.exists():
            return 0
        
        loaded_count = 0
        
        # Load Python files
        for python_file in directory_path.glob("*.py"):
            if python_file.name.startswith("_"):
                continue  # Skip private files
            
            success = await self.load_plugin(str(python_file))
            if success:
                loaded_count += 1
        
        # Load plugin packages
        for plugin_dir in directory_path.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith("_"):
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    success = await self.load_plugin(str(plugin_dir))
                    if success:
                        loaded_count += 1
        
        return loaded_count
    
    async def initialize_all_plugins(self) -> int:
        """Initialize all loaded plugins."""
        
        initialized_count = 0
        
        for plugin_name, plugin_info in self.plugins.items():
            if plugin_info.status == PluginStatus.LOADED:
                success = await self.initialize_plugin(plugin_name)
                if success:
                    initialized_count += 1
        
        return initialized_count
    
    async def activate_all_plugins(self) -> int:
        """Activate all initialized plugins."""
        
        activated_count = 0
        
        # Sort by dependencies (simplified)
        plugin_names = list(self.plugins.keys())
        
        for plugin_name in plugin_names:
            plugin_info = self.plugins.get(plugin_name)
            if plugin_info and plugin_info.status == PluginStatus.INITIALIZED:
                success = await self.activate_plugin(plugin_name)
                if success:
                    activated_count += 1
        
        return activated_count
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """List all plugins, optionally filtered by status."""
        
        plugins = list(self.plugins.values())
        
        if status_filter:
            plugins = [p for p in plugins if p.status == status_filter]
        
        return plugins
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get detailed information about a plugin."""
        
        return self.plugins.get(plugin_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        
        status_counts = {}
        for status in PluginStatus:
            status_counts[status.value] = sum(
                1 for p in self.plugins.values() if p.status == status
            )
        
        type_counts = {}
        for plugin_type in PluginType:
            type_counts[plugin_type.value] = len(self.plugins_by_type[plugin_type])
        
        capability_counts = {}
        for capability in PluginCapability:
            capability_counts[capability.value] = len(self.plugins_by_capability[capability])
        
        return {
            "total_plugins": len(self.plugins),
            "plugins_by_status": status_counts,
            "plugins_by_type": type_counts,
            "plugins_by_capability": capability_counts,
            "active_hooks": len(self.event_hooks)
        }
    
    async def shutdown(self) -> None:
        """Shutdown all plugins and the plugin manager."""
        
        # Deactivate all active plugins
        for plugin_name, plugin_info in list(self.plugins.items()):
            if plugin_info.status == PluginStatus.ACTIVE:
                await self.deactivate_plugin(plugin_name)
        
        # Shutdown all initialized plugins
        for plugin_name, plugin_info in list(self.plugins.items()):
            if plugin_info.plugin_instance and plugin_info.status in [PluginStatus.INITIALIZED, PluginStatus.ERROR]:
                await plugin_info.plugin_instance.shutdown()
        
        # Clear all data
        self.plugins.clear()
        self.event_hooks.clear()
        for plugin_list in self.plugins_by_type.values():
            plugin_list.clear()
        for plugin_list in self.plugins_by_capability.values():
            plugin_list.clear()
    
    async def _auto_load_plugins(self) -> None:
        """Automatically load plugins from configured directories."""
        
        for directory in self.plugin_directories:
            await self.load_plugins_from_directory(directory)
        
        await self.initialize_all_plugins()
        await self.activate_all_plugins()
    
    async def _register_plugin_hooks(self, plugin: BasePlugin) -> None:
        """Register hooks from a plugin."""
        
        # Get hook methods from plugin
        from ._base_plugin import get_plugin_hooks
        hooks = get_plugin_hooks(plugin)
        
        for event, hook_methods in hooks.items():
            if event not in self.event_hooks:
                self.event_hooks[event] = []
            
            # Add plugin to event hooks if not already present
            if plugin not in self.event_hooks[event]:
                self.event_hooks[event].append(plugin)
    
    async def _unregister_plugin_hooks(self, plugin: BasePlugin) -> None:
        """Unregister hooks from a plugin."""
        
        for event, plugins in self.event_hooks.items():
            if plugin in plugins:
                plugins.remove(plugin)
    
    async def _handle_plugin_error(self, plugin_name: str, error_message: str) -> None:
        """Handle plugin errors."""
        
        if plugin_name in self.plugins:
            self.plugins[plugin_name].status = PluginStatus.ERROR
            self.plugins[plugin_name].error_message = error_message
        
        # Log error (in a real implementation, use proper logging)
        print(f"Plugin Error [{plugin_name}]: {error_message}")
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        
        if not self.config_file:
            return
        
        config_path = Path(self.config_file)
        if not config_path.exists():
            return
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Failed to load plugin config: {e}")
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        
        if not self.config_file:
            return False
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save plugin config: {e}")
            return False
