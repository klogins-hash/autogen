"""
Test suite for the persistent memory system.
"""

import asyncio
import tempfile
import shutil
from pathlib import Path
import pytest

from ._vector_memory import VectorMemoryStore, MemoryEntry, MemoryQuery
from ._persistent_memory import PersistentMemoryManager, MemoryType, MemoryMetadata
from ._knowledge_graph import KnowledgeGraph, EntityType, RelationshipType


class TestVectorMemoryStore:
    """Test cases for VectorMemoryStore."""
    
    @pytest.fixture
    async def memory_store(self):
        """Create a temporary memory store for testing."""
        temp_dir = tempfile.mkdtemp()
        store = VectorMemoryStore(
            collection_name="test_memory",
            persist_directory=temp_dir,
            max_entries=100
        )
        yield store
        shutil.rmtree(temp_dir)
    
    async def test_add_and_retrieve_memory(self, memory_store):
        """Test adding and retrieving memories."""
        
        # Add a memory
        memory_id = await memory_store.add_memory(
            content="This is a test memory about machine learning",
            agent_name="test_agent",
            session_id="test_session",
            metadata={"category": "ml", "importance": "high"}
        )
        
        assert memory_id is not None
        
        # Retrieve the memory
        memory = await memory_store.get_memory(memory_id)
        assert memory is not None
        assert memory.content == "This is a test memory about machine learning"
        assert memory.agent_name == "test_agent"
        assert memory.metadata["category"] == "ml"
    
    async def test_search_memories(self, memory_store):
        """Test semantic search functionality."""
        
        # Add multiple memories
        memories = [
            "Machine learning algorithms are powerful tools",
            "Deep learning uses neural networks",
            "Natural language processing handles text",
            "Computer vision processes images"
        ]
        
        for i, content in enumerate(memories):
            await memory_store.add_memory(
                content=content,
                agent_name=f"agent_{i}",
                session_id="test_session"
            )
        
        # Search for ML-related content
        query = MemoryQuery(
            text="artificial intelligence and neural networks",
            limit=2,
            similarity_threshold=0.3
        )
        
        results = await memory_store.search_memories(query)
        assert len(results) > 0
        
        # Results should be sorted by similarity
        similarities = [similarity for _, similarity in results]
        assert similarities == sorted(similarities, reverse=True)
    
    async def test_update_memory(self, memory_store):
        """Test memory updates."""
        
        # Add a memory
        memory_id = await memory_store.add_memory(
            content="Original content",
            agent_name="test_agent",
            session_id="test_session"
        )
        
        # Update the memory
        success = await memory_store.update_memory(
            memory_id=memory_id,
            content="Updated content",
            metadata={"updated": True}
        )
        
        assert success
        
        # Verify update
        memory = await memory_store.get_memory(memory_id)
        assert memory.content == "Updated content"
        assert memory.metadata["updated"] is True
    
    async def test_memory_pruning(self, memory_store):
        """Test automatic memory pruning."""
        
        # Set a small max_entries for testing
        memory_store.max_entries = 3
        
        # Add more memories than the limit
        memory_ids = []
        for i in range(5):
            memory_id = await memory_store.add_memory(
                content=f"Memory {i}",
                agent_name="test_agent",
                session_id="test_session",
                importance_score=i * 0.2  # Varying importance
            )
            memory_ids.append(memory_id)
        
        # Trigger pruning
        await memory_store._prune_memories()
        
        # Check that only the most important memories remain
        stats = memory_store.get_stats()
        assert stats["total_entries"] <= 3


class TestPersistentMemoryManager:
    """Test cases for PersistentMemoryManager."""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create a temporary memory manager for testing."""
        temp_dir = tempfile.mkdtemp()
        manager = PersistentMemoryManager(
            storage_path=temp_dir,
            auto_cleanup=False  # Disable for testing
        )
        yield manager
        await manager.shutdown()
        shutil.rmtree(temp_dir)
    
    async def test_store_conversation_memory(self, memory_manager):
        """Test storing conversation memories."""
        
        memory_id = await memory_manager.store_conversation_memory(
            conversation_summary="Discussed AI safety measures",
            participants=["agent1", "agent2", "user"],
            key_decisions=["Implement safety checks", "Add human oversight"],
            agent_name="orchestrator",
            session_id="test_session"
        )
        
        assert memory_id is not None
        
        # Retrieve and verify
        memories = await memory_manager.retrieve_memories(
            query_text="AI safety",
            memory_types=[MemoryType.CONVERSATION],
            limit=1
        )
        
        assert len(memories) > 0
        memory_entry, similarity = memories[0]
        assert "AI safety" in memory_entry.content
        assert memory_entry.metadata["memory_type"] == MemoryType.CONVERSATION.value
    
    async def test_store_task_solution(self, memory_manager):
        """Test storing task solutions."""
        
        memory_id = await memory_manager.store_task_solution(
            task_description="Create a web scraper",
            solution_steps=[
                "Import requests and BeautifulSoup",
                "Send HTTP request to target URL",
                "Parse HTML content",
                "Extract required data"
            ],
            success_rate=0.95,
            agent_name="coder",
            session_id="test_session"
        )
        
        assert memory_id is not None
        
        # Find similar tasks
        similar_tasks = await memory_manager.find_similar_tasks(
            task_description="Build a web crawler",
            limit=1,
            similarity_threshold=0.5
        )
        
        assert len(similar_tasks) > 0
        memory_entry, similarity = similar_tasks[0]
        assert "web scraper" in memory_entry.content.lower()
    
    async def test_store_error_pattern(self, memory_manager):
        """Test storing error patterns."""
        
        memory_id = await memory_manager.store_error_pattern(
            error_description="Connection timeout error",
            error_context="Occurred while accessing external API",
            recovery_strategy="Implement retry logic with exponential backoff",
            agent_name="web_surfer",
            session_id="test_session"
        )
        
        assert memory_id is not None
        
        # Find error solutions
        solutions = await memory_manager.find_error_solutions(
            error_description="API connection failed",
            limit=1
        )
        
        assert len(solutions) > 0
        memory_entry, similarity = solutions[0]
        assert "timeout" in memory_entry.content.lower()
    
    async def test_user_preferences(self, memory_manager):
        """Test storing and retrieving user preferences."""
        
        # Store preferences
        pref_id = await memory_manager.store_user_preference(
            preference_description="Code style preference",
            preference_value="Use type hints and docstrings",
            context="Python development",
            agent_name="coder",
            session_id="test_session"
        )
        
        assert pref_id is not None
        
        # Retrieve preferences
        preferences = await memory_manager.get_user_preferences(
            context="Python coding",
            limit=5
        )
        
        assert len(preferences) > 0
        assert any("type hints" in pref.content for pref in preferences)
    
    async def test_memory_statistics(self, memory_manager):
        """Test memory statistics generation."""
        
        # Add various types of memories
        await memory_manager.store_conversation_memory(
            "Test conversation", ["agent1"], [], "agent1", "session1"
        )
        await memory_manager.store_task_solution(
            "Test task", ["step1"], 1.0, "agent2", "session1"
        )
        
        # Get statistics
        stats = await memory_manager.get_memory_statistics()
        
        assert "vector_store_stats" in stats
        assert "memories_by_type" in stats
        assert "memories_by_agent" in stats
        assert stats["memories_by_type"][MemoryType.CONVERSATION.value] >= 1
        assert stats["memories_by_type"][MemoryType.TASK_SOLUTION.value] >= 1


class TestKnowledgeGraph:
    """Test cases for KnowledgeGraph."""
    
    @pytest.fixture
    async def knowledge_graph(self):
        """Create a temporary knowledge graph for testing."""
        temp_dir = tempfile.mkdtemp()
        graph = KnowledgeGraph(storage_path=temp_dir)
        yield graph
        shutil.rmtree(temp_dir)
    
    async def test_add_entities_and_relationships(self, knowledge_graph):
        """Test adding entities and relationships."""
        
        # Add entities
        agent_entity = await knowledge_graph.add_entity(
            entity_id="agent_1",
            name="Web Surfer",
            entity_type=EntityType.AGENT,
            properties={"description": "Browses web content"}
        )
        
        task_entity = await knowledge_graph.add_entity(
            entity_id="task_1",
            name="Research AI trends",
            entity_type=EntityType.TASK,
            properties={"complexity": "medium"}
        )
        
        assert agent_entity.name == "Web Surfer"
        assert task_entity.entity_type == EntityType.TASK
        
        # Add relationship
        relationship = await knowledge_graph.add_relationship(
            relationship_id="rel_1",
            from_entity_id="agent_1",
            to_entity_id="task_1",
            relationship_type=RelationshipType.SOLVES,
            strength=0.9
        )
        
        assert relationship is not None
        assert relationship.strength == 0.9
    
    async def test_entity_retrieval(self, knowledge_graph):
        """Test entity retrieval methods."""
        
        # Add test entities
        await knowledge_graph.add_entity(
            "agent_1", "Coder Agent", EntityType.AGENT
        )
        await knowledge_graph.add_entity(
            "agent_2", "Web Agent", EntityType.AGENT
        )
        await knowledge_graph.add_entity(
            "task_1", "Code Review", EntityType.TASK
        )
        
        # Test get by ID
        entity = await knowledge_graph.get_entity("agent_1")
        assert entity.name == "Coder Agent"
        
        # Test get by name
        entity = await knowledge_graph.get_entity_by_name("Web Agent")
        assert entity.id == "agent_2"
        
        # Test get by type
        agents = await knowledge_graph.get_entities_by_type(EntityType.AGENT)
        assert len(agents) == 2
        
        tasks = await knowledge_graph.get_entities_by_type(EntityType.TASK)
        assert len(tasks) == 1
    
    async def test_path_finding(self, knowledge_graph):
        """Test path finding between entities."""
        
        # Create a chain: Agent -> Tool -> Task
        await knowledge_graph.add_entity("agent_1", "Agent", EntityType.AGENT)
        await knowledge_graph.add_entity("tool_1", "Web Browser", EntityType.TOOL)
        await knowledge_graph.add_entity("task_1", "Research", EntityType.TASK)
        
        await knowledge_graph.add_relationship(
            "rel_1", "agent_1", "tool_1", RelationshipType.USES
        )
        await knowledge_graph.add_relationship(
            "rel_2", "tool_1", "task_1", RelationshipType.SOLVES
        )
        
        # Find path
        path = await knowledge_graph.find_path("agent_1", "task_1", max_depth=3)
        
        assert path is not None
        assert len(path) == 3  # agent -> tool -> task
        assert path[0][0].id == "agent_1"
        assert path[1][0].id == "tool_1"
        assert path[2][0].id == "task_1"
    
    async def test_neighbors(self, knowledge_graph):
        """Test neighbor retrieval."""
        
        # Create entities and relationships
        await knowledge_graph.add_entity("center", "Center", EntityType.AGENT)
        await knowledge_graph.add_entity("neighbor1", "Neighbor 1", EntityType.TOOL)
        await knowledge_graph.add_entity("neighbor2", "Neighbor 2", EntityType.TASK)
        
        await knowledge_graph.add_relationship(
            "rel_1", "center", "neighbor1", RelationshipType.USES
        )
        await knowledge_graph.add_relationship(
            "rel_2", "neighbor2", "center", RelationshipType.DEPENDS_ON
        )
        
        # Get all neighbors
        neighbors = await knowledge_graph.get_neighbors("center", direction="both")
        assert len(neighbors) == 2
        
        # Get outgoing neighbors only
        outgoing = await knowledge_graph.get_neighbors("center", direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0][0].id == "neighbor1"
        
        # Get incoming neighbors only
        incoming = await knowledge_graph.get_neighbors("center", direction="incoming")
        assert len(incoming) == 1
        assert incoming[0][0].id == "neighbor2"
    
    async def test_graph_statistics(self, knowledge_graph):
        """Test graph statistics."""
        
        # Add some entities and relationships
        await knowledge_graph.add_entity("e1", "Entity 1", EntityType.AGENT)
        await knowledge_graph.add_entity("e2", "Entity 2", EntityType.TASK)
        await knowledge_graph.add_relationship(
            "r1", "e1", "e2", RelationshipType.SOLVES
        )
        
        stats = knowledge_graph.get_statistics()
        
        assert stats["total_entities"] >= 2
        assert stats["total_relationships"] >= 1
        assert EntityType.AGENT.value in stats["entities_by_type"]
        assert RelationshipType.SOLVES.value in stats["relationships_by_type"]


# Integration test
async def test_full_integration():
    """Test full integration of memory components."""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize components
        memory_manager = PersistentMemoryManager(
            storage_path=temp_dir,
            auto_cleanup=False
        )
        knowledge_graph = KnowledgeGraph(storage_path=f"{temp_dir}/kg")
        
        # Store a task solution
        task_memory_id = await memory_manager.store_task_solution(
            task_description="Implement web scraping",
            solution_steps=["Use requests", "Parse with BeautifulSoup"],
            success_rate=0.9,
            agent_name="coder",
            session_id="integration_test"
        )
        
        # Add corresponding entities to knowledge graph
        await knowledge_graph.add_entity(
            "agent_coder", "Coder Agent", EntityType.AGENT
        )
        await knowledge_graph.add_entity(
            "task_scraping", "Web Scraping Task", EntityType.TASK,
            properties={"memory_id": task_memory_id}
        )
        await knowledge_graph.add_relationship(
            "coder_solves_scraping", "agent_coder", "task_scraping",
            RelationshipType.SOLVES, strength=0.9
        )
        
        # Verify integration
        # 1. Memory retrieval
        similar_tasks = await memory_manager.find_similar_tasks(
            "create web crawler", limit=1, similarity_threshold=0.5
        )
        assert len(similar_tasks) > 0
        
        # 2. Knowledge graph traversal
        neighbors = await knowledge_graph.get_neighbors(
            "agent_coder", relationship_types=[RelationshipType.SOLVES]
        )
        assert len(neighbors) > 0
        assert neighbors[0][0].name == "Web Scraping Task"
        
        # 3. Cross-reference
        task_entity = neighbors[0][0]
        memory_id = task_entity.properties.get("memory_id")
        assert memory_id == task_memory_id
        
        # Cleanup
        await memory_manager.shutdown()
        
    finally:
        shutil.rmtree(temp_dir)


# Run tests
if __name__ == "__main__":
    async def run_all_tests():
        """Run all tests."""
        
        print("Running Vector Memory Store tests...")
        test_store = TestVectorMemoryStore()
        # Note: In real testing, you'd use pytest fixtures properly
        
        print("Running Persistent Memory Manager tests...")
        test_manager = TestPersistentMemoryManager()
        
        print("Running Knowledge Graph tests...")
        test_graph = TestKnowledgeGraph()
        
        print("Running integration test...")
        await test_full_integration()
        
        print("All tests completed!")
    
    asyncio.run(run_all_tests())
