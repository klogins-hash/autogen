"""
Persistent memory management system for AutoGen agents.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from ._vector_memory import VectorMemoryStore, MemoryEntry, MemoryQuery


class MemoryType(Enum):
    """Types of memories that can be stored."""
    CONVERSATION = "conversation"
    TASK_SOLUTION = "task_solution"
    ERROR_PATTERN = "error_pattern"
    USER_PREFERENCE = "user_preference"
    AGENT_BEHAVIOR = "agent_behavior"
    KNOWLEDGE_FACT = "knowledge_fact"
    WORKFLOW_PATTERN = "workflow_pattern"
    CUSTOM = "custom"


@dataclass
class MemoryMetadata:
    """Metadata for memory entries."""
    memory_type: MemoryType
    tags: Set[str] = field(default_factory=set)
    source_agent: str = ""
    related_task_id: str = ""
    confidence_score: float = 1.0
    validation_status: str = "unvalidated"  # unvalidated, validated, deprecated
    created_by: str = ""
    last_updated: float = field(default_factory=time.time)


class PersistentMemoryManager:
    """Manages persistent memory across agent sessions with intelligent organization."""
    
    def __init__(
        self,
        storage_path: str = "./agent_memory",
        vector_store_name: str = "persistent_memory",
        max_memories_per_type: int = 1000,
        auto_cleanup: bool = True,
        cleanup_interval: float = 3600.0  # 1 hour
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = VectorMemoryStore(
            collection_name=vector_store_name,
            persist_directory=str(self.storage_path / "vectors"),
            max_entries=max_memories_per_type * len(MemoryType)
        )
        
        self.max_memories_per_type = max_memories_per_type
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Memory organization
        self.memory_index: Dict[MemoryType, List[str]] = {mt: [] for mt in MemoryType}
        self.tag_index: Dict[str, Set[str]] = {}
        self.agent_memories: Dict[str, Set[str]] = {}
        
        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Load existing index
        asyncio.create_task(self._load_memory_index())
        
        if auto_cleanup:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        agent_name: str,
        metadata: Optional[MemoryMetadata] = None,
        importance_score: float = 1.0,
        session_id: str = ""
    ) -> str:
        """Store a new memory with metadata."""
        
        if metadata is None:
            metadata = MemoryMetadata(memory_type=memory_type, source_agent=agent_name)
        
        # Prepare vector store metadata
        vector_metadata = {
            "memory_type": memory_type.value,
            "tags": list(metadata.tags),
            "source_agent": metadata.source_agent,
            "related_task_id": metadata.related_task_id,
            "confidence_score": metadata.confidence_score,
            "validation_status": metadata.validation_status,
            "created_by": metadata.created_by,
            "last_updated": metadata.last_updated
        }
        
        # Store in vector database
        memory_id = await self.vector_store.add_memory(
            content=content,
            agent_name=agent_name,
            session_id=session_id,
            metadata=vector_metadata,
            importance_score=importance_score
        )
        
        # Update indices
        self.memory_index[memory_type].append(memory_id)
        
        for tag in metadata.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(memory_id)
        
        if agent_name not in self.agent_memories:
            self.agent_memories[agent_name] = set()
        self.agent_memories[agent_name].add(memory_id)
        
        # Save updated index
        await self._save_memory_index()
        
        return memory_id
    
    async def retrieve_memories(
        self,
        query_text: str,
        memory_types: Optional[List[MemoryType]] = None,
        agent_name: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve memories based on semantic similarity and filters."""
        
        # Build filters
        filters = {}
        
        if memory_types:
            # ChromaDB doesn't support OR queries directly, so we'll filter post-retrieval
            pass
        
        if agent_name:
            filters["agent_name"] = agent_name
        
        # Create query
        query = MemoryQuery(
            text=query_text,
            filters=filters,
            limit=limit * 2,  # Get more results to filter
            similarity_threshold=similarity_threshold
        )
        
        # Search vector store
        results = await self.vector_store.search_memories(query)
        
        # Apply additional filters
        filtered_results = []
        for entry, similarity in results:
            # Filter by memory type
            if memory_types:
                entry_type = MemoryType(entry.metadata.get("memory_type", "custom"))
                if entry_type not in memory_types:
                    continue
            
            # Filter by tags
            if tags:
                entry_tags = set(entry.metadata.get("tags", []))
                if not tags.intersection(entry_tags):
                    continue
            
            filtered_results.append((entry, similarity))
            
            if len(filtered_results) >= limit:
                break
        
        return filtered_results
    
    async def get_agent_memories(
        self,
        agent_name: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Get all memories for a specific agent."""
        
        memories = await self.vector_store.get_memories_by_agent(agent_name, limit)
        
        if memory_type:
            memories = [
                m for m in memories 
                if m.metadata.get("memory_type") == memory_type.value
            ]
        
        return memories
    
    async def store_conversation_memory(
        self,
        conversation_summary: str,
        participants: List[str],
        key_decisions: List[str],
        agent_name: str,
        session_id: str = "",
        importance_score: float = 1.0
    ) -> str:
        """Store a conversation memory with structured metadata."""
        
        metadata = MemoryMetadata(
            memory_type=MemoryType.CONVERSATION,
            tags={"conversation", "summary"} | set(participants),
            source_agent=agent_name,
            confidence_score=importance_score
        )
        
        # Create structured content
        content = f"""Conversation Summary: {conversation_summary}
Participants: {', '.join(participants)}
Key Decisions: {'; '.join(key_decisions)}"""
        
        return await self.store_memory(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            agent_name=agent_name,
            metadata=metadata,
            importance_score=importance_score,
            session_id=session_id
        )
    
    async def store_task_solution(
        self,
        task_description: str,
        solution_steps: List[str],
        success_rate: float,
        agent_name: str,
        session_id: str = "",
        related_task_id: str = ""
    ) -> str:
        """Store a successful task solution for future reference."""
        
        metadata = MemoryMetadata(
            memory_type=MemoryType.TASK_SOLUTION,
            tags={"task", "solution", "successful"},
            source_agent=agent_name,
            related_task_id=related_task_id,
            confidence_score=success_rate
        )
        
        content = f"""Task: {task_description}
Solution Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(solution_steps))}
Success Rate: {success_rate:.2%}"""
        
        return await self.store_memory(
            content=content,
            memory_type=MemoryType.TASK_SOLUTION,
            agent_name=agent_name,
            metadata=metadata,
            importance_score=success_rate,
            session_id=session_id
        )
    
    async def store_error_pattern(
        self,
        error_description: str,
        error_context: str,
        recovery_strategy: str,
        agent_name: str,
        session_id: str = ""
    ) -> str:
        """Store an error pattern and its recovery strategy."""
        
        metadata = MemoryMetadata(
            memory_type=MemoryType.ERROR_PATTERN,
            tags={"error", "pattern", "recovery"},
            source_agent=agent_name,
            confidence_score=0.8  # Error patterns are important but not always applicable
        )
        
        content = f"""Error: {error_description}
Context: {error_context}
Recovery Strategy: {recovery_strategy}"""
        
        return await self.store_memory(
            content=content,
            memory_type=MemoryType.ERROR_PATTERN,
            agent_name=agent_name,
            metadata=metadata,
            importance_score=0.8,
            session_id=session_id
        )
    
    async def store_user_preference(
        self,
        preference_description: str,
        preference_value: str,
        context: str,
        agent_name: str,
        session_id: str = ""
    ) -> str:
        """Store a user preference for personalization."""
        
        metadata = MemoryMetadata(
            memory_type=MemoryType.USER_PREFERENCE,
            tags={"user", "preference", "personalization"},
            source_agent=agent_name,
            confidence_score=1.0  # User preferences are highly important
        )
        
        content = f"""Preference: {preference_description}
Value: {preference_value}
Context: {context}"""
        
        return await self.store_memory(
            content=content,
            memory_type=MemoryType.USER_PREFERENCE,
            agent_name=agent_name,
            metadata=metadata,
            importance_score=1.0,
            session_id=session_id
        )
    
    async def find_similar_tasks(
        self,
        task_description: str,
        limit: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Tuple[MemoryEntry, float]]:
        """Find similar tasks that have been solved before."""
        
        return await self.retrieve_memories(
            query_text=task_description,
            memory_types=[MemoryType.TASK_SOLUTION],
            limit=limit,
            similarity_threshold=similarity_threshold
        )
    
    async def find_error_solutions(
        self,
        error_description: str,
        limit: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[MemoryEntry, float]]:
        """Find solutions for similar errors."""
        
        return await self.retrieve_memories(
            query_text=error_description,
            memory_types=[MemoryType.ERROR_PATTERN],
            limit=limit,
            similarity_threshold=similarity_threshold
        )
    
    async def get_user_preferences(
        self,
        context: Optional[str] = None,
        limit: int = 20
    ) -> List[MemoryEntry]:
        """Get user preferences, optionally filtered by context."""
        
        if context:
            results = await self.retrieve_memories(
                query_text=context,
                memory_types=[MemoryType.USER_PREFERENCE],
                limit=limit,
                similarity_threshold=0.5
            )
            return [entry for entry, _ in results]
        else:
            # Get all user preferences
            memories = await self.vector_store.get_memories_by_agent("", limit * 2)
            return [
                m for m in memories 
                if m.metadata.get("memory_type") == MemoryType.USER_PREFERENCE.value
            ][:limit]
    
    async def validate_memory(self, memory_id: str, is_valid: bool) -> bool:
        """Mark a memory as validated or deprecated."""
        
        status = "validated" if is_valid else "deprecated"
        
        return await self.vector_store.update_memory(
            memory_id=memory_id,
            metadata={"validation_status": status, "last_updated": time.time()}
        )
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        
        vector_stats = self.vector_store.get_stats()
        
        # Count memories by type
        type_counts = {}
        for memory_type, memory_ids in self.memory_index.items():
            type_counts[memory_type.value] = len(memory_ids)
        
        # Count memories by agent
        agent_counts = {agent: len(memories) for agent, memories in self.agent_memories.items()}
        
        # Count by tags
        tag_counts = {tag: len(memories) for tag, memories in self.tag_index.items()}
        
        return {
            "vector_store_stats": vector_stats,
            "memories_by_type": type_counts,
            "memories_by_agent": agent_counts,
            "memories_by_tag": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "total_tags": len(self.tag_index),
            "total_agents": len(self.agent_memories)
        }
    
    async def _load_memory_index(self) -> None:
        """Load memory index from disk."""
        
        index_file = self.storage_path / "memory_index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                
                # Load memory type index
                for type_name, memory_ids in data.get("memory_index", {}).items():
                    try:
                        memory_type = MemoryType(type_name)
                        self.memory_index[memory_type] = memory_ids
                    except ValueError:
                        pass  # Skip unknown memory types
                
                # Load tag index
                for tag, memory_ids in data.get("tag_index", {}).items():
                    self.tag_index[tag] = set(memory_ids)
                
                # Load agent memories
                for agent, memory_ids in data.get("agent_memories", {}).items():
                    self.agent_memories[agent] = set(memory_ids)
                    
            except Exception:
                pass  # Start with empty index if loading fails
    
    async def _save_memory_index(self) -> None:
        """Save memory index to disk."""
        
        index_file = self.storage_path / "memory_index.json"
        
        try:
            data = {
                "memory_index": {mt.value: ids for mt, ids in self.memory_index.items()},
                "tag_index": {tag: list(ids) for tag, ids in self.tag_index.items()},
                "agent_memories": {agent: list(ids) for agent, ids in self.agent_memories.items()},
                "last_updated": time.time()
            }
            
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception:
            pass  # Continue if saving fails
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old and low-value memories."""
        
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_memories()
            except asyncio.CancelledError:
                break
            except Exception:
                continue  # Continue cleanup loop even if individual cleanup fails
    
    async def _cleanup_memories(self) -> None:
        """Clean up old and low-value memories."""
        
        # Clean up each memory type separately
        for memory_type in MemoryType:
            memory_ids = self.memory_index[memory_type]
            
            if len(memory_ids) > self.max_memories_per_type:
                # Get memories for this type
                memories_to_evaluate = []
                
                for memory_id in memory_ids:
                    memory = await self.vector_store.get_memory(memory_id)
                    if memory:
                        memories_to_evaluate.append(memory)
                
                # Sort by importance and age
                memories_to_evaluate.sort(
                    key=lambda m: (
                        m.importance_score * (1 + m.access_count),
                        -m.timestamp  # Newer is better
                    ),
                    reverse=True
                )
                
                # Keep only the best memories
                to_keep = memories_to_evaluate[:self.max_memories_per_type]
                to_remove = memories_to_evaluate[self.max_memories_per_type:]
                
                # Remove excess memories
                for memory in to_remove:
                    await self.vector_store.delete_memory(memory.id)
                    
                    # Update indices
                    if memory.id in memory_ids:
                        memory_ids.remove(memory.id)
                    
                    for tag_memories in self.tag_index.values():
                        tag_memories.discard(memory.id)
                    
                    for agent_memories in self.agent_memories.values():
                        agent_memories.discard(memory.id)
        
        # Save updated index
        await self._save_memory_index()
    
    async def shutdown(self) -> None:
        """Shutdown the memory manager."""
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self._save_memory_index()
