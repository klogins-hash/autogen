"""
Knowledge graph implementation for structured relationship storage and reasoning.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    AGENT = "agent"
    TASK = "task"
    CONCEPT = "concept"
    TOOL = "tool"
    USER = "user"
    ERROR = "error"
    SOLUTION = "solution"
    WORKFLOW = "workflow"
    CUSTOM = "custom"


class RelationshipType(Enum):
    """Types of relationships between entities."""
    USES = "uses"
    SOLVES = "solves"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    HANDLES = "handles"
    COLLABORATES_WITH = "collaborates_with"
    IMPROVES = "improves"
    CUSTOM = "custom"


@dataclass
class Entity:
    """Entity in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    importance_score: float = 1.0
    access_count: int = 0


@dataclass
class Relationship:
    """Relationship between entities."""
    id: str
    from_entity_id: str
    to_entity_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class GraphQuery:
    """Query for knowledge graph traversal."""
    start_entity_id: Optional[str] = None
    entity_types: Optional[List[EntityType]] = None
    relationship_types: Optional[List[RelationshipType]] = None
    max_depth: int = 3
    min_strength: float = 0.1
    limit: int = 100


class KnowledgeGraph:
    """Knowledge graph for storing and querying structured relationships."""
    
    def __init__(self, storage_path: str = "./knowledge_graph"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Indices for fast lookup
        self.entity_by_type: Dict[EntityType, Set[str]] = {et: set() for et in EntityType}
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.relationships_from: Dict[str, Set[str]] = {}  # entity_id -> relationship_ids
        self.relationships_to: Dict[str, Set[str]] = {}  # entity_id -> relationship_ids
        
        # Load existing data
        asyncio.create_task(self._load_graph())
    
    async def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: EntityType,
        properties: Optional[Dict[str, Any]] = None,
        importance_score: float = 1.0
    ) -> Entity:
        """Add an entity to the knowledge graph."""
        
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            importance_score=importance_score
        )
        
        self.entities[entity_id] = entity
        self.entity_by_type[entity_type].add(entity_id)
        self.entity_by_name[name.lower()] = entity_id
        
        await self._save_graph()
        return entity
    
    async def add_relationship(
        self,
        relationship_id: str,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None,
        strength: float = 1.0,
        confidence: float = 1.0
    ) -> Optional[Relationship]:
        """Add a relationship between entities."""
        
        # Verify entities exist
        if from_entity_id not in self.entities or to_entity_id not in self.entities:
            return None
        
        relationship = Relationship(
            id=relationship_id,
            from_entity_id=from_entity_id,
            to_entity_id=to_entity_id,
            relationship_type=relationship_type,
            properties=properties or {},
            strength=strength,
            confidence=confidence
        )
        
        self.relationships[relationship_id] = relationship
        
        # Update indices
        if from_entity_id not in self.relationships_from:
            self.relationships_from[from_entity_id] = set()
        self.relationships_from[from_entity_id].add(relationship_id)
        
        if to_entity_id not in self.relationships_to:
            self.relationships_to[to_entity_id] = set()
        self.relationships_to[to_entity_id].add(relationship_id)
        
        await self._save_graph()
        return relationship
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        entity = self.entities.get(entity_id)
        if entity:
            entity.access_count += 1
            entity.updated_at = time.time()
        return entity
    
    async def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        entity_id = self.entity_by_name.get(name.lower())
        if entity_id:
            return await self.get_entity(entity_id)
        return None
    
    async def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_by_type.get(entity_type, set())
        entities = []
        
        for entity_id in entity_ids:
            entity = await self.get_entity(entity_id)
            if entity:
                entities.append(entity)
        
        return entities
    
    async def get_relationships(
        self,
        from_entity_id: Optional[str] = None,
        to_entity_id: Optional[str] = None,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get relationships matching criteria."""
        
        candidate_ids = set(self.relationships.keys())
        
        # Filter by from_entity
        if from_entity_id:
            from_relationships = self.relationships_from.get(from_entity_id, set())
            candidate_ids &= from_relationships
        
        # Filter by to_entity
        if to_entity_id:
            to_relationships = self.relationships_to.get(to_entity_id, set())
            candidate_ids &= to_relationships
        
        # Filter by type and return
        relationships = []
        for rel_id in candidate_ids:
            relationship = self.relationships.get(rel_id)
            if relationship and (not relationship_type or relationship.relationship_type == relationship_type):
                relationships.append(relationship)
        
        return relationships
    
    async def find_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> Optional[List[Tuple[Entity, Relationship]]]:
        """Find a path between two entities using BFS."""
        
        if start_entity_id == end_entity_id:
            start_entity = await self.get_entity(start_entity_id)
            return [(start_entity, None)] if start_entity else None
        
        visited = set()
        queue = [(start_entity_id, [])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id in visited or len(path) >= max_depth:
                continue
            
            visited.add(current_id)
            
            # Get outgoing relationships
            outgoing_rels = self.relationships_from.get(current_id, set())
            
            for rel_id in outgoing_rels:
                relationship = self.relationships.get(rel_id)
                if not relationship:
                    continue
                
                # Filter by relationship type if specified
                if relationship_types and relationship.relationship_type not in relationship_types:
                    continue
                
                next_entity_id = relationship.to_entity_id
                
                if next_entity_id == end_entity_id:
                    # Found path
                    final_path = []
                    for entity_id, rel in path:
                        entity = await self.get_entity(entity_id)
                        if entity:
                            final_path.append((entity, rel))
                    
                    # Add final entities
                    current_entity = await self.get_entity(current_id)
                    end_entity = await self.get_entity(end_entity_id)
                    
                    if current_entity and end_entity:
                        final_path.append((current_entity, relationship))
                        final_path.append((end_entity, None))
                        return final_path
                
                if next_entity_id not in visited:
                    new_path = path + [(current_id, relationship)]
                    queue.append((next_entity_id, new_path))
        
        return None  # No path found
    
    async def get_neighbors(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Tuple[Entity, Relationship]]:
        """Get neighboring entities."""
        
        neighbors = []
        
        # Outgoing relationships
        if direction in ["outgoing", "both"]:
            outgoing_rels = self.relationships_from.get(entity_id, set())
            for rel_id in outgoing_rels:
                relationship = self.relationships.get(rel_id)
                if relationship and (not relationship_types or relationship.relationship_type in relationship_types):
                    neighbor = await self.get_entity(relationship.to_entity_id)
                    if neighbor:
                        neighbors.append((neighbor, relationship))
        
        # Incoming relationships
        if direction in ["incoming", "both"]:
            incoming_rels = self.relationships_to.get(entity_id, set())
            for rel_id in incoming_rels:
                relationship = self.relationships.get(rel_id)
                if relationship and (not relationship_types or relationship.relationship_type in relationship_types):
                    neighbor = await self.get_entity(relationship.from_entity_id)
                    if neighbor:
                        neighbors.append((neighbor, relationship))
        
        return neighbors
    
    async def query_graph(self, query: GraphQuery) -> Dict[str, Any]:
        """Execute a complex graph query."""
        
        results = {
            "entities": [],
            "relationships": [],
            "paths": []
        }
        
        # Start with specific entity or all entities of specified types
        start_entities = []
        
        if query.start_entity_id:
            entity = await self.get_entity(query.start_entity_id)
            if entity:
                start_entities = [entity]
        elif query.entity_types:
            for entity_type in query.entity_types:
                entities = await self.get_entities_by_type(entity_type)
                start_entities.extend(entities)
        else:
            start_entities = list(self.entities.values())
        
        # Traverse from start entities
        visited_entities = set()
        visited_relationships = set()
        
        for start_entity in start_entities[:query.limit]:
            await self._traverse_from_entity(
                start_entity.id,
                query,
                0,
                visited_entities,
                visited_relationships,
                results
            )
        
        return results
    
    async def _traverse_from_entity(
        self,
        entity_id: str,
        query: GraphQuery,
        depth: int,
        visited_entities: Set[str],
        visited_relationships: Set[str],
        results: Dict[str, Any]
    ) -> None:
        """Recursively traverse the graph from an entity."""
        
        if depth >= query.max_depth or entity_id in visited_entities:
            return
        
        visited_entities.add(entity_id)
        entity = await self.get_entity(entity_id)
        
        if entity:
            results["entities"].append(entity)
        
        # Get outgoing relationships
        outgoing_rels = self.relationships_from.get(entity_id, set())
        
        for rel_id in outgoing_rels:
            if rel_id in visited_relationships:
                continue
            
            relationship = self.relationships.get(rel_id)
            if not relationship:
                continue
            
            # Filter by relationship type and strength
            if query.relationship_types and relationship.relationship_type not in query.relationship_types:
                continue
            
            if relationship.strength < query.min_strength:
                continue
            
            visited_relationships.add(rel_id)
            results["relationships"].append(relationship)
            
            # Continue traversal
            await self._traverse_from_entity(
                relationship.to_entity_id,
                query,
                depth + 1,
                visited_entities,
                visited_relationships,
                results
            )
    
    async def get_similar_entities(
        self,
        entity_id: str,
        similarity_threshold: float = 0.5,
        limit: int = 10
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to the given entity."""
        
        # Get entities connected by "similar_to" relationships
        similar_relationships = await self.get_relationships(
            from_entity_id=entity_id,
            relationship_type=RelationshipType.SIMILAR_TO
        )
        
        similar_entities = []
        
        for relationship in similar_relationships:
            if relationship.strength >= similarity_threshold:
                entity = await self.get_entity(relationship.to_entity_id)
                if entity:
                    similar_entities.append((entity, relationship.strength))
        
        # Sort by similarity strength
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        
        return similar_entities[:limit]
    
    async def update_entity(
        self,
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> bool:
        """Update an entity's properties."""
        
        entity = self.entities.get(entity_id)
        if not entity:
            return False
        
        if properties:
            entity.properties.update(properties)
        
        if importance_score is not None:
            entity.importance_score = importance_score
        
        entity.updated_at = time.time()
        await self._save_graph()
        return True
    
    async def update_relationship(
        self,
        relationship_id: str,
        properties: Optional[Dict[str, Any]] = None,
        strength: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """Update a relationship's properties."""
        
        relationship = self.relationships.get(relationship_id)
        if not relationship:
            return False
        
        if properties:
            relationship.properties.update(properties)
        
        if strength is not None:
            relationship.strength = strength
        
        if confidence is not None:
            relationship.confidence = confidence
        
        await self._save_graph()
        return True
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        
        entity = self.entities.get(entity_id)
        if not entity:
            return False
        
        # Delete all relationships involving this entity
        relationships_to_delete = set()
        
        # Outgoing relationships
        outgoing = self.relationships_from.get(entity_id, set())
        relationships_to_delete.update(outgoing)
        
        # Incoming relationships
        incoming = self.relationships_to.get(entity_id, set())
        relationships_to_delete.update(incoming)
        
        # Delete relationships
        for rel_id in relationships_to_delete:
            await self.delete_relationship(rel_id)
        
        # Delete entity
        del self.entities[entity_id]
        self.entity_by_type[entity.entity_type].discard(entity_id)
        
        # Remove from name index
        for name, eid in list(self.entity_by_name.items()):
            if eid == entity_id:
                del self.entity_by_name[name]
                break
        
        await self._save_graph()
        return True
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship."""
        
        relationship = self.relationships.get(relationship_id)
        if not relationship:
            return False
        
        # Remove from indices
        self.relationships_from.get(relationship.from_entity_id, set()).discard(relationship_id)
        self.relationships_to.get(relationship.to_entity_id, set()).discard(relationship_id)
        
        # Delete relationship
        del self.relationships[relationship_id]
        
        await self._save_graph()
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        
        entity_counts = {}
        for entity_type, entity_ids in self.entity_by_type.items():
            entity_counts[entity_type.value] = len(entity_ids)
        
        relationship_counts = {}
        for relationship in self.relationships.values():
            rel_type = relationship.relationship_type.value
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entities_by_type": entity_counts,
            "relationships_by_type": relationship_counts,
            "avg_connections_per_entity": len(self.relationships) * 2 / len(self.entities) if self.entities else 0
        }
    
    async def _load_graph(self) -> None:
        """Load graph from disk."""
        
        entities_file = self.storage_path / "entities.json"
        relationships_file = self.storage_path / "relationships.json"
        
        # Load entities
        if entities_file.exists():
            try:
                with open(entities_file, 'r') as f:
                    entities_data = json.load(f)
                
                for entity_data in entities_data:
                    entity = Entity(**entity_data)
                    entity.entity_type = EntityType(entity.entity_type)
                    
                    self.entities[entity.id] = entity
                    self.entity_by_type[entity.entity_type].add(entity.id)
                    self.entity_by_name[entity.name.lower()] = entity.id
                    
            except Exception:
                pass
        
        # Load relationships
        if relationships_file.exists():
            try:
                with open(relationships_file, 'r') as f:
                    relationships_data = json.load(f)
                
                for rel_data in relationships_data:
                    relationship = Relationship(**rel_data)
                    relationship.relationship_type = RelationshipType(relationship.relationship_type)
                    
                    self.relationships[relationship.id] = relationship
                    
                    # Update indices
                    if relationship.from_entity_id not in self.relationships_from:
                        self.relationships_from[relationship.from_entity_id] = set()
                    self.relationships_from[relationship.from_entity_id].add(relationship.id)
                    
                    if relationship.to_entity_id not in self.relationships_to:
                        self.relationships_to[relationship.to_entity_id] = set()
                    self.relationships_to[relationship.to_entity_id].add(relationship.id)
                    
            except Exception:
                pass
    
    async def _save_graph(self) -> None:
        """Save graph to disk."""
        
        try:
            # Save entities
            entities_data = []
            for entity in self.entities.values():
                entity_dict = {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "properties": entity.properties,
                    "created_at": entity.created_at,
                    "updated_at": entity.updated_at,
                    "importance_score": entity.importance_score,
                    "access_count": entity.access_count
                }
                entities_data.append(entity_dict)
            
            with open(self.storage_path / "entities.json", 'w') as f:
                json.dump(entities_data, f, indent=2)
            
            # Save relationships
            relationships_data = []
            for relationship in self.relationships.values():
                rel_dict = {
                    "id": relationship.id,
                    "from_entity_id": relationship.from_entity_id,
                    "to_entity_id": relationship.to_entity_id,
                    "relationship_type": relationship.relationship_type.value,
                    "properties": relationship.properties,
                    "strength": relationship.strength,
                    "created_at": relationship.created_at,
                    "confidence": relationship.confidence
                }
                relationships_data.append(rel_dict)
            
            with open(self.storage_path / "relationships.json", 'w') as f:
                json.dump(relationships_data, f, indent=2)
                
        except Exception:
            pass  # Continue if saving fails
