"""
Vector-based memory storage for semantic similarity and retrieval.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingProvider(Enum):
    """Available embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class MemoryEntry:
    """Entry in vector memory store."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    agent_name: str = ""
    session_id: str = ""
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class MemoryQuery:
    """Query for memory retrieval."""
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True


class VectorMemoryStore:
    """Vector-based memory store with semantic search capabilities."""
    
    def __init__(
        self,
        collection_name: str = "agent_memory",
        embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        max_entries: int = 10000
    ):
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.max_entries = max_entries
        
        # Initialize embedding model
        self.embedder = self._initialize_embedder()
        
        # Initialize vector database
        self.client = None
        self.collection = None
        self._initialize_database()
        
    def _initialize_embedder(self):
        """Initialize the embedding model."""
        if self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            return SentenceTransformer(self.embedding_model)
        
        elif self.embedding_provider == EmbeddingProvider.OPENAI:
            # Would integrate with OpenAI embeddings API
            return None
        
        else:
            return None
    
    def _initialize_database(self):
        """Initialize the vector database."""
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not available. Install with: pip install chromadb")
        
        if self.persist_directory:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Collection might already exist
            self.collection = self.client.get_collection(name=self.collection_name)
    
    async def add_memory(
        self,
        content: str,
        agent_name: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: float = 1.0
    ) -> str:
        """Add a memory entry."""
        
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Create memory entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            agent_name=agent_name,
            session_id=session_id,
            importance_score=importance_score
        )
        
        # Prepare data for ChromaDB
        chroma_metadata = {
            "agent_name": agent_name,
            "session_id": session_id,
            "importance_score": importance_score,
            "timestamp": entry.timestamp,
            "access_count": 0
        }
        
        # Add custom metadata
        if metadata:
            chroma_metadata.update(metadata)
        
        # Add to vector database
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[chroma_metadata]
        )
        
        # Check if we need to prune old entries
        await self._prune_if_needed()
        
        return memory_id
    
    async def search_memories(
        self,
        query: MemoryQuery
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories."""
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query.text)
        
        # Prepare where clause for filtering
        where_clause = {}
        if query.filters:
            where_clause.update(query.filters)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=query.limit,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert results to MemoryEntry objects
        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                
                # Convert distance to similarity score (cosine similarity)
                similarity = 1.0 - distance
                
                # Filter by similarity threshold
                if similarity >= query.similarity_threshold:
                    entry = MemoryEntry(
                        id=memory_id,
                        content=content,
                        metadata=metadata,
                        agent_name=metadata.get("agent_name", ""),
                        session_id=metadata.get("session_id", ""),
                        importance_score=metadata.get("importance_score", 1.0),
                        timestamp=metadata.get("timestamp", time.time()),
                        access_count=metadata.get("access_count", 0)
                    )
                    
                    memories.append((entry, similarity))
                    
                    # Update access count
                    await self._update_access_count(memory_id)
        
        return memories
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        
        try:
            result = self.collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"]
            )
            
            if result["ids"] and result["ids"][0]:
                content = result["documents"][0]
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                
                entry = MemoryEntry(
                    id=memory_id,
                    content=content,
                    metadata=metadata,
                    agent_name=metadata.get("agent_name", ""),
                    session_id=metadata.get("session_id", ""),
                    importance_score=metadata.get("importance_score", 1.0),
                    timestamp=metadata.get("timestamp", time.time()),
                    access_count=metadata.get("access_count", 0)
                )
                
                await self._update_access_count(memory_id)
                return entry
                
        except Exception:
            pass
        
        return None
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> bool:
        """Update an existing memory."""
        
        try:
            # Get current memory
            current = await self.get_memory(memory_id)
            if not current:
                return False
            
            # Prepare updates
            new_content = content or current.content
            new_metadata = current.metadata.copy()
            if metadata:
                new_metadata.update(metadata)
            
            if importance_score is not None:
                new_metadata["importance_score"] = importance_score
            
            # Generate new embedding if content changed
            if content and content != current.content:
                new_embedding = await self._generate_embedding(new_content)
                
                # Delete old entry and add new one
                self.collection.delete(ids=[memory_id])
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[new_embedding],
                    documents=[new_content],
                    metadatas=[new_metadata]
                )
            else:
                # Update metadata only
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[new_metadata]
                )
            
            return True
            
        except Exception:
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    async def get_memories_by_agent(
        self,
        agent_name: str,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Get all memories for a specific agent."""
        
        query = MemoryQuery(
            text="",  # Empty query to get all
            filters={"agent_name": agent_name},
            limit=limit,
            similarity_threshold=0.0
        )
        
        # Use a generic query to get all memories for the agent
        try:
            results = self.collection.get(
                where={"agent_name": agent_name},
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    content = results["documents"][i]
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    
                    entry = MemoryEntry(
                        id=memory_id,
                        content=content,
                        metadata=metadata,
                        agent_name=metadata.get("agent_name", ""),
                        session_id=metadata.get("session_id", ""),
                        importance_score=metadata.get("importance_score", 1.0),
                        timestamp=metadata.get("timestamp", time.time()),
                        access_count=metadata.get("access_count", 0)
                    )
                    
                    memories.append(entry)
            
            return memories
            
        except Exception:
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        
        if self.embedding_provider == EmbeddingProvider.SENTENCE_TRANSFORMERS and self.embedder:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.embedder.encode(text).tolist()
            )
            return embedding
        
        # Fallback: simple hash-based embedding (not recommended for production)
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to float values (very basic embedding)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0
            embedding.append(val)
        
        # Pad to standard size
        while len(embedding) < 384:  # Standard size for all-MiniLM-L6-v2
            embedding.append(0.0)
        
        return embedding[:384]
    
    async def _update_access_count(self, memory_id: str) -> None:
        """Update access count for a memory."""
        
        try:
            current = self.collection.get(
                ids=[memory_id],
                include=["metadatas"]
            )
            
            if current["metadatas"] and current["metadatas"][0]:
                metadata = current["metadatas"][0]
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_accessed"] = time.time()
                
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
        except Exception:
            pass
    
    async def _prune_if_needed(self) -> None:
        """Prune old memories if we exceed max_entries."""
        
        try:
            count = self.collection.count()
            
            if count > self.max_entries:
                # Get oldest memories with lowest importance
                all_memories = self.collection.get(
                    include=["metadatas"],
                    limit=count
                )
                
                if all_memories["ids"] and all_memories["metadatas"]:
                    # Sort by importance and age
                    memory_scores = []
                    for i, memory_id in enumerate(all_memories["ids"]):
                        metadata = all_memories["metadatas"][i]
                        importance = metadata.get("importance_score", 1.0)
                        age = time.time() - metadata.get("timestamp", time.time())
                        access_count = metadata.get("access_count", 0)
                        
                        # Lower score = more likely to be pruned
                        score = importance * (1 + access_count) / (1 + age / 86400)  # Age in days
                        memory_scores.append((memory_id, score))
                    
                    # Sort by score and remove lowest scoring memories
                    memory_scores.sort(key=lambda x: x[1])
                    to_remove = count - self.max_entries
                    
                    ids_to_delete = [item[0] for item in memory_scores[:to_remove]]
                    self.collection.delete(ids=ids_to_delete)
                    
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        
        try:
            count = self.collection.count()
            
            # Get sample of memories for stats
            sample = self.collection.get(
                limit=min(1000, count),
                include=["metadatas"]
            )
            
            if sample["metadatas"]:
                agents = set()
                sessions = set()
                total_importance = 0
                total_access = 0
                
                for metadata in sample["metadatas"]:
                    agents.add(metadata.get("agent_name", ""))
                    sessions.add(metadata.get("session_id", ""))
                    total_importance += metadata.get("importance_score", 1.0)
                    total_access += metadata.get("access_count", 0)
                
                return {
                    "total_memories": count,
                    "unique_agents": len(agents),
                    "unique_sessions": len(sessions),
                    "avg_importance": total_importance / len(sample["metadatas"]),
                    "avg_access_count": total_access / len(sample["metadatas"]),
                    "max_entries": self.max_entries,
                    "utilization": count / self.max_entries
                }
            
            return {"total_memories": count, "max_entries": self.max_entries}
            
        except Exception:
            return {"error": "Unable to get stats"}
    
    async def clear_memories(self, agent_name: Optional[str] = None) -> int:
        """Clear memories, optionally filtered by agent."""
        
        try:
            if agent_name:
                # Delete memories for specific agent
                memories = self.collection.get(
                    where={"agent_name": agent_name},
                    include=["documents"]
                )
                
                if memories["ids"]:
                    self.collection.delete(ids=memories["ids"])
                    return len(memories["ids"])
                return 0
            else:
                # Clear all memories
                count = self.collection.count()
                self.collection.delete()
                return count
                
        except Exception:
            return 0
