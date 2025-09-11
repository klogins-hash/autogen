"""
Message caching system for reducing redundant API calls and improving performance.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from autogen_core.models import LLMMessage


class CacheStrategy(Enum):
    """Available caching strategies."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    SEMANTIC = "semantic"  # Semantic similarity based


@dataclass
class CacheEntry:
    """Cache entry containing response and metadata."""
    key: str
    response: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp


class MessageCache:
    """Intelligent caching system for LLM responses."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour default
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU tracking
        
    def _generate_cache_key(
        self, 
        messages: List[LLMMessage], 
        model_params: Dict[str, Any] = None
    ) -> str:
        """Generate a unique cache key for the request."""
        # Create a deterministic representation of the messages
        message_data = []
        for msg in messages:
            msg_dict = {
                "type": msg.__class__.__name__,
                "content": str(msg.content),
                "source": getattr(msg, 'source', None)
            }
            message_data.append(msg_dict)
        
        # Include model parameters in the key
        cache_data = {
            "messages": message_data,
            "params": model_params or {}
        }
        
        # Create hash of the serialized data
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def get(
        self, 
        messages: List[LLMMessage], 
        model_params: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Retrieve cached response if available."""
        cache_key = self._generate_cache_key(messages, model_params)
        
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # Check TTL expiration
        if self.strategy == CacheStrategy.TTL:
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._remove_entry(cache_key)
                return None
        
        # Update access tracking
        entry.access_count += 1
        entry.last_accessed = time.time()
        
        # Update LRU order
        if self.strategy == CacheStrategy.LRU:
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
        
        return entry.response
    
    def put(
        self, 
        messages: List[LLMMessage], 
        response: Any,
        model_params: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Store response in cache."""
        cache_key = self._generate_cache_key(messages, model_params)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            response=response,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict_entry()
        
        # Store the entry
        self.cache[cache_key] = entry
        
        # Update access order for LRU
        if self.strategy == CacheStrategy.LRU:
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
    
    def _evict_entry(self) -> None:
        """Evict an entry based on the caching strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order[0]
                self._remove_entry(lru_key)
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove oldest entry
            oldest_key = min(
                self.cache.keys(), 
                key=lambda k: self.cache[k].timestamp
            )
            self._remove_entry(oldest_key)
        
        else:
            # Default: remove oldest
            oldest_key = min(
                self.cache.keys(), 
                key=lambda k: self.cache[k].timestamp
            )
            self._remove_entry(oldest_key)
    
    def _remove_entry(self, cache_key: str) -> None:
        """Remove an entry from the cache."""
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count of removed entries."""
        if self.strategy != CacheStrategy.TTL:
            return 0
        
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        if self.cache:
            avg_age = time.time() - sum(
                entry.timestamp for entry in self.cache.values()
            ) / total_entries
            
            most_accessed = max(
                self.cache.values(), 
                key=lambda e: e.access_count
            )
        else:
            avg_age = 0
            most_accessed = None
        
        return {
            "total_entries": total_entries,
            "max_size": self.max_size,
            "utilization": total_entries / self.max_size if self.max_size > 0 else 0,
            "total_accesses": total_accesses,
            "average_age_seconds": avg_age,
            "most_accessed_key": most_accessed.key if most_accessed else None,
            "most_accessed_count": most_accessed.access_count if most_accessed else 0,
            "strategy": self.strategy.value
        }
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        matching_keys = [
            key for key in self.cache.keys()
            if pattern in key
        ]
        
        for key in matching_keys:
            self._remove_entry(key)
        
        return len(matching_keys)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()


class SemanticMessageCache(MessageCache):
    """Cache with semantic similarity matching for related queries."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.8
    ):
        super().__init__(max_size, ttl_seconds, CacheStrategy.SEMANTIC)
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache: Dict[str, List[float]] = {}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple keyword-based similarity for now
        # In production, this would use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def get_similar(
        self, 
        messages: List[LLMMessage], 
        model_params: Dict[str, Any] = None
    ) -> Optional[Tuple[Any, float]]:
        """Find similar cached response with similarity score."""
        query_text = " ".join(
            str(msg.content) for msg in messages 
            if isinstance(msg.content, str)
        )
        
        best_match = None
        best_similarity = 0.0
        
        for cache_key, entry in self.cache.items():
            # Reconstruct text from cache key (simplified)
            cached_text = cache_key  # In practice, store original text
            similarity = self._calculate_similarity(query_text, cached_text)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry.response
        
        if best_match:
            return best_match, best_similarity
        
        return None
