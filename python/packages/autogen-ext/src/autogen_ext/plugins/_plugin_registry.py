"""
Plugin registry for managing plugin metadata and discovery.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from ._base_plugin import PluginType, PluginCapability


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    capabilities: Set[PluginCapability]
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    homepage: str = ""
    repository: str = ""
    license: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    download_count: int = 0
    rating: float = 0.0
    verified: bool = False


class PluginRegistry:
    """Registry for plugin metadata and discovery."""
    
    def __init__(self, registry_file: str = "./plugin_registry.json"):
        self.registry_file = Path(registry_file)
        self.plugins: Dict[str, PluginMetadata] = {}
        self.tags_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[PluginType, Set[str]] = {pt: set() for pt in PluginType}
        self.capability_index: Dict[PluginCapability, Set[str]] = {pc: set() for pc in PluginCapability}
        
        self._load_registry()
    
    def register_plugin(self, metadata: PluginMetadata) -> bool:
        """Register a plugin in the registry."""
        
        try:
            # Update timestamp
            metadata.updated_at = time.time()
            
            # Store plugin
            self.plugins[metadata.name] = metadata
            
            # Update indices
            self._update_indices(metadata)
            
            # Save to file
            self._save_registry()
            
            return True
            
        except Exception as e:
            print(f"Failed to register plugin {metadata.name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin from the registry."""
        
        if plugin_name not in self.plugins:
            return False
        
        try:
            metadata = self.plugins[plugin_name]
            
            # Remove from indices
            self._remove_from_indices(metadata)
            
            # Remove plugin
            del self.plugins[plugin_name]
            
            # Save to file
            self._save_registry()
            
            return True
            
        except Exception as e:
            print(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        return self.plugins.get(plugin_name)
    
    def search_plugins(
        self,
        query: Optional[str] = None,
        plugin_type: Optional[PluginType] = None,
        capabilities: Optional[Set[PluginCapability]] = None,
        tags: Optional[Set[str]] = None,
        verified_only: bool = False
    ) -> List[PluginMetadata]:
        """Search for plugins based on criteria."""
        
        results = list(self.plugins.values())
        
        # Filter by type
        if plugin_type:
            results = [p for p in results if p.plugin_type == plugin_type]
        
        # Filter by capabilities
        if capabilities:
            results = [p for p in results if capabilities.issubset(p.capabilities)]
        
        # Filter by tags
        if tags:
            results = [p for p in results if tags.intersection(p.tags)]
        
        # Filter by verified status
        if verified_only:
            results = [p for p in results if p.verified]
        
        # Filter by query (simple text search)
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if (query_lower in p.name.lower() or
                    query_lower in p.description.lower() or
                    any(query_lower in tag.lower() for tag in p.tags))
            ]
        
        # Sort by rating and download count
        results.sort(key=lambda p: (p.rating, p.download_count), reverse=True)
        
        return results
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginMetadata]:
        """Get all plugins of a specific type."""
        plugin_names = self.type_index.get(plugin_type, set())
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def get_plugins_by_capability(self, capability: PluginCapability) -> List[PluginMetadata]:
        """Get all plugins with a specific capability."""
        plugin_names = self.capability_index.get(capability, set())
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def get_plugins_by_tag(self, tag: str) -> List[PluginMetadata]:
        """Get all plugins with a specific tag."""
        plugin_names = self.tags_index.get(tag.lower(), set())
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def get_popular_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """Get most popular plugins by download count."""
        plugins = list(self.plugins.values())
        plugins.sort(key=lambda p: p.download_count, reverse=True)
        return plugins[:limit]
    
    def get_top_rated_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """Get top-rated plugins."""
        plugins = [p for p in self.plugins.values() if p.rating > 0]
        plugins.sort(key=lambda p: p.rating, reverse=True)
        return plugins[:limit]
    
    def get_recent_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """Get recently added plugins."""
        plugins = list(self.plugins.values())
        plugins.sort(key=lambda p: p.created_at, reverse=True)
        return plugins[:limit]
    
    def increment_download_count(self, plugin_name: str) -> bool:
        """Increment download count for a plugin."""
        
        if plugin_name not in self.plugins:
            return False
        
        self.plugins[plugin_name].download_count += 1
        self._save_registry()
        return True
    
    def update_rating(self, plugin_name: str, rating: float) -> bool:
        """Update plugin rating."""
        
        if plugin_name not in self.plugins:
            return False
        
        if not 0 <= rating <= 5:
            return False
        
        self.plugins[plugin_name].rating = rating
        self.plugins[plugin_name].updated_at = time.time()
        self._save_registry()
        return True
    
    def verify_plugin(self, plugin_name: str, verified: bool = True) -> bool:
        """Mark a plugin as verified or unverified."""
        
        if plugin_name not in self.plugins:
            return False
        
        self.plugins[plugin_name].verified = verified
        self.plugins[plugin_name].updated_at = time.time()
        self._save_registry()
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        
        total_plugins = len(self.plugins)
        verified_plugins = sum(1 for p in self.plugins.values() if p.verified)
        
        type_counts = {}
        for plugin_type in PluginType:
            type_counts[plugin_type.value] = len(self.type_index[plugin_type])
        
        capability_counts = {}
        for capability in PluginCapability:
            capability_counts[capability.value] = len(self.capability_index[capability])
        
        # Top tags
        tag_counts = {tag: len(plugins) for tag, plugins in self.tags_index.items()}
        top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            "total_plugins": total_plugins,
            "verified_plugins": verified_plugins,
            "verification_rate": verified_plugins / total_plugins if total_plugins > 0 else 0,
            "plugins_by_type": type_counts,
            "plugins_by_capability": capability_counts,
            "top_tags": top_tags,
            "total_downloads": sum(p.download_count for p in self.plugins.values()),
            "average_rating": sum(p.rating for p in self.plugins.values() if p.rating > 0) / 
                            len([p for p in self.plugins.values() if p.rating > 0]) if 
                            any(p.rating > 0 for p in self.plugins.values()) else 0
        }
    
    def export_registry(self, export_file: str) -> bool:
        """Export registry to a file."""
        
        try:
            export_data = {
                "plugins": {
                    name: self._metadata_to_dict(metadata)
                    for name, metadata in self.plugins.items()
                },
                "exported_at": time.time(),
                "version": "1.0"
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, import_file: str, merge: bool = True) -> bool:
        """Import registry from a file."""
        
        try:
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            if not merge:
                self.plugins.clear()
                self._clear_indices()
            
            for name, plugin_data in import_data.get("plugins", {}).items():
                metadata = self._dict_to_metadata(plugin_data)
                if metadata:
                    self.plugins[name] = metadata
                    self._update_indices(metadata)
            
            self._save_registry()
            return True
            
        except Exception as e:
            print(f"Failed to import registry: {e}")
            return False
    
    def _update_indices(self, metadata: PluginMetadata) -> None:
        """Update search indices for a plugin."""
        
        # Type index
        self.type_index[metadata.plugin_type].add(metadata.name)
        
        # Capability index
        for capability in metadata.capabilities:
            self.capability_index[capability].add(metadata.name)
        
        # Tags index
        for tag in metadata.tags:
            tag_lower = tag.lower()
            if tag_lower not in self.tags_index:
                self.tags_index[tag_lower] = set()
            self.tags_index[tag_lower].add(metadata.name)
    
    def _remove_from_indices(self, metadata: PluginMetadata) -> None:
        """Remove a plugin from search indices."""
        
        # Type index
        self.type_index[metadata.plugin_type].discard(metadata.name)
        
        # Capability index
        for capability in metadata.capabilities:
            self.capability_index[capability].discard(metadata.name)
        
        # Tags index
        for tag in metadata.tags:
            tag_lower = tag.lower()
            if tag_lower in self.tags_index:
                self.tags_index[tag_lower].discard(metadata.name)
                if not self.tags_index[tag_lower]:
                    del self.tags_index[tag_lower]
    
    def _clear_indices(self) -> None:
        """Clear all search indices."""
        
        self.tags_index.clear()
        for plugin_set in self.type_index.values():
            plugin_set.clear()
        for plugin_set in self.capability_index.values():
            plugin_set.clear()
    
    def _load_registry(self) -> None:
        """Load registry from file."""
        
        if not self.registry_file.exists():
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            for name, plugin_data in data.get("plugins", {}).items():
                metadata = self._dict_to_metadata(plugin_data)
                if metadata:
                    self.plugins[name] = metadata
                    self._update_indices(metadata)
                    
        except Exception as e:
            print(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        
        try:
            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "plugins": {
                    name: self._metadata_to_dict(metadata)
                    for name, metadata in self.plugins.items()
                },
                "last_updated": time.time(),
                "version": "1.0"
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save registry: {e}")
    
    def _metadata_to_dict(self, metadata: PluginMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        
        data = asdict(metadata)
        
        # Convert enums to strings
        data["plugin_type"] = metadata.plugin_type.value
        data["capabilities"] = [cap.value for cap in metadata.capabilities]
        data["tags"] = list(metadata.tags)
        
        return data
    
    def _dict_to_metadata(self, data: Dict[str, Any]) -> Optional[PluginMetadata]:
        """Convert dictionary to metadata object."""
        
        try:
            # Convert strings back to enums
            plugin_type = PluginType(data["plugin_type"])
            capabilities = {PluginCapability(cap) for cap in data["capabilities"]}
            tags = set(data.get("tags", []))
            
            return PluginMetadata(
                name=data["name"],
                version=data["version"],
                description=data["description"],
                author=data.get("author", ""),
                plugin_type=plugin_type,
                capabilities=capabilities,
                dependencies=data.get("dependencies", []),
                requirements=data.get("requirements", []),
                tags=tags,
                homepage=data.get("homepage", ""),
                repository=data.get("repository", ""),
                license=data.get("license", ""),
                created_at=data.get("created_at", time.time()),
                updated_at=data.get("updated_at", time.time()),
                download_count=data.get("download_count", 0),
                rating=data.get("rating", 0.0),
                verified=data.get("verified", False)
            )
            
        except Exception as e:
            print(f"Failed to convert dict to metadata: {e}")
            return None
