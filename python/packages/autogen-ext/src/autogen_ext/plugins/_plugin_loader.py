"""
Plugin loader for dynamically loading AutoGen plugins.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ._base_plugin import BasePlugin


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails."""
    pass


class PluginLoader:
    """Loads plugins from various sources."""
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
    
    async def load_plugin_class(self, plugin_path: str) -> Optional[Type[BasePlugin]]:
        """Load a plugin class from a file or module path."""
        
        try:
            # Handle different path types
            if plugin_path.endswith('.py'):
                return await self._load_from_file(plugin_path)
            elif '.' in plugin_path:
                return await self._load_from_module(plugin_path)
            else:
                # Try as directory with __init__.py
                return await self._load_from_directory(plugin_path)
                
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin from {plugin_path}: {e}")
    
    async def _load_from_file(self, file_path: str) -> Optional[Type[BasePlugin]]:
        """Load plugin from a Python file."""
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise PluginLoadError(f"Plugin file not found: {file_path}")
        
        # Create module spec
        module_name = f"plugin_{file_path_obj.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if not spec or not spec.loader:
            raise PluginLoadError(f"Could not create module spec for {file_path}")
        
        # Load module
        module = importlib.util.module_from_spec(spec)
        self._loaded_modules[module_name] = module
        
        # Add to sys.modules to support imports
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise PluginLoadError(f"Failed to execute module {file_path}: {e}")
        
        # Find plugin class
        return self._find_plugin_class(module)
    
    async def _load_from_module(self, module_path: str) -> Optional[Type[BasePlugin]]:
        """Load plugin from a module path (e.g., 'my_package.my_plugin')."""
        
        try:
            module = importlib.import_module(module_path)
            self._loaded_modules[module_path] = module
            return self._find_plugin_class(module)
        except ImportError as e:
            raise PluginLoadError(f"Could not import module {module_path}: {e}")
    
    async def _load_from_directory(self, directory_path: str) -> Optional[Type[BasePlugin]]:
        """Load plugin from a directory with __init__.py."""
        
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise PluginLoadError(f"Plugin directory not found: {directory_path}")
        
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            raise PluginLoadError(f"No __init__.py found in {directory_path}")
        
        return await self._load_from_file(str(init_file))
    
    def _find_plugin_class(self, module: Any) -> Optional[Type[BasePlugin]]:
        """Find the plugin class in a module."""
        
        plugin_classes = []
        
        # Look for classes that inherit from BasePlugin
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj != BasePlugin and 
                issubclass(obj, BasePlugin) and 
                obj.__module__ == module.__name__):
                plugin_classes.append(obj)
        
        if not plugin_classes:
            raise PluginLoadError(f"No plugin class found in module {module.__name__}")
        
        if len(plugin_classes) > 1:
            # If multiple classes, look for one with 'Plugin' in the name
            named_plugins = [cls for cls in plugin_classes if 'Plugin' in cls.__name__]
            if len(named_plugins) == 1:
                return named_plugins[0]
            else:
                raise PluginLoadError(
                    f"Multiple plugin classes found in module {module.__name__}: "
                    f"{[cls.__name__ for cls in plugin_classes]}"
                )
        
        return plugin_classes[0]
    
    def unload_module(self, module_name: str) -> bool:
        """Unload a previously loaded module."""
        
        try:
            if module_name in self._loaded_modules:
                del self._loaded_modules[module_name]
            
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return True
        except Exception:
            return False
    
    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded module names."""
        return list(self._loaded_modules.keys())
    
    def clear_cache(self) -> None:
        """Clear the module cache."""
        for module_name in list(self._loaded_modules.keys()):
            self.unload_module(module_name)
