"""
Integration example showing how to use the persistent memory system with Magentic-One agents.
"""

import asyncio
from typing import List, Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ChatCompletionClient

from ._persistent_memory import PersistentMemoryManager, MemoryType, MemoryMetadata
from ._knowledge_graph import KnowledgeGraph, EntityType, RelationshipType


class MemoryEnhancedAgent(AssistantAgent):
    """Agent enhanced with persistent memory capabilities."""
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        memory_manager: PersistentMemoryManager,
        knowledge_graph: KnowledgeGraph,
        **kwargs
    ):
        super().__init__(name=name, model_client=model_client, **kwargs)
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.session_id = f"session_{int(asyncio.get_event_loop().time())}"
        
        # Initialize agent entity in knowledge graph
        asyncio.create_task(self._initialize_agent_entity())
    
    async def _initialize_agent_entity(self):
        """Initialize this agent as an entity in the knowledge graph."""
        await self.knowledge_graph.add_entity(
            entity_id=f"agent_{self.name}",
            name=self.name,
            entity_type=EntityType.AGENT,
            properties={
                "description": getattr(self, "description", ""),
                "capabilities": getattr(self, "tools", []),
                "created_at": asyncio.get_event_loop().time()
            }
        )
    
    async def on_messages(self, messages: List[TextMessage], cancellation_token=None) -> TextMessage:
        """Enhanced message processing with memory integration."""
        
        # Retrieve relevant memories before processing
        await self._retrieve_relevant_memories(messages)
        
        # Process messages normally
        response = await super().on_messages(messages, cancellation_token)
        
        # Store conversation memory after processing
        await self._store_conversation_memory(messages, response)
        
        return response
    
    async def _retrieve_relevant_memories(self, messages: List[TextMessage]):
        """Retrieve and inject relevant memories into the conversation context."""
        
        # Extract key topics from recent messages
        recent_content = " ".join([msg.content for msg in messages[-3:]])
        
        # Search for relevant memories
        relevant_memories = await self.memory_manager.retrieve_memories(
            query_text=recent_content,
            agent_name=self.name,
            limit=5,
            similarity_threshold=0.7
        )
        
        if relevant_memories:
            memory_context = "Relevant past experiences:\n"
            for memory_entry, similarity in relevant_memories:
                memory_context += f"- {memory_entry.content[:200]}... (similarity: {similarity:.2f})\n"
            
            # Add memory context to system message (implementation depends on your agent setup)
            # This is a simplified example - you'd integrate this into your agent's prompt
            self._add_memory_context(memory_context)
    
    async def _store_conversation_memory(self, messages: List[TextMessage], response: TextMessage):
        """Store important conversation elements in memory."""
        
        # Extract participants
        participants = list(set([msg.source for msg in messages if hasattr(msg, 'source')]))
        participants.append(self.name)
        
        # Create conversation summary
        conversation_text = "\n".join([f"{getattr(msg, 'source', 'Unknown')}: {msg.content}" for msg in messages[-5:]])
        conversation_text += f"\n{self.name}: {response.content}"
        
        # Store conversation memory
        await self.memory_manager.store_conversation_memory(
            conversation_summary=conversation_text[:500],  # Truncate for storage
            participants=participants,
            key_decisions=[],  # Could be extracted using NLP
            agent_name=self.name,
            session_id=self.session_id
        )
    
    async def store_task_success(
        self,
        task_description: str,
        solution_steps: List[str],
        success_rate: float = 1.0
    ):
        """Store a successful task completion for future reference."""
        
        memory_id = await self.memory_manager.store_task_solution(
            task_description=task_description,
            solution_steps=solution_steps,
            success_rate=success_rate,
            agent_name=self.name,
            session_id=self.session_id
        )
        
        # Also store in knowledge graph
        task_entity_id = f"task_{memory_id}"
        await self.knowledge_graph.add_entity(
            entity_id=task_entity_id,
            name=task_description[:100],
            entity_type=EntityType.TASK,
            properties={
                "description": task_description,
                "solution_steps": solution_steps,
                "success_rate": success_rate,
                "memory_id": memory_id
            }
        )
        
        # Create relationship between agent and task
        await self.knowledge_graph.add_relationship(
            relationship_id=f"solved_{memory_id}",
            from_entity_id=f"agent_{self.name}",
            to_entity_id=task_entity_id,
            relationship_type=RelationshipType.SOLVES,
            strength=success_rate
        )
    
    async def store_error_recovery(
        self,
        error_description: str,
        error_context: str,
        recovery_strategy: str
    ):
        """Store error recovery information."""
        
        memory_id = await self.memory_manager.store_error_pattern(
            error_description=error_description,
            error_context=error_context,
            recovery_strategy=recovery_strategy,
            agent_name=self.name,
            session_id=self.session_id
        )
        
        # Store in knowledge graph
        error_entity_id = f"error_{memory_id}"
        await self.knowledge_graph.add_entity(
            entity_id=error_entity_id,
            name=error_description[:100],
            entity_type=EntityType.ERROR,
            properties={
                "description": error_description,
                "context": error_context,
                "recovery_strategy": recovery_strategy,
                "memory_id": memory_id
            }
        )
        
        # Create relationship
        await self.knowledge_graph.add_relationship(
            relationship_id=f"handles_{memory_id}",
            from_entity_id=f"agent_{self.name}",
            to_entity_id=error_entity_id,
            relationship_type=RelationshipType.HANDLES
        )
    
    async def find_similar_tasks(self, task_description: str) -> List[Dict[str, Any]]:
        """Find similar tasks that have been solved before."""
        
        similar_memories = await self.memory_manager.find_similar_tasks(
            task_description=task_description,
            limit=5,
            similarity_threshold=0.8
        )
        
        results = []
        for memory_entry, similarity in similar_memories:
            results.append({
                "memory_id": memory_entry.id,
                "content": memory_entry.content,
                "similarity": similarity,
                "agent": memory_entry.agent_name,
                "timestamp": memory_entry.timestamp
            })
        
        return results
    
    async def get_collaboration_history(self, other_agent_name: str) -> List[Dict[str, Any]]:
        """Get history of collaboration with another agent."""
        
        # Search for conversations involving both agents
        memories = await self.memory_manager.retrieve_memories(
            query_text=f"collaboration {other_agent_name}",
            memory_types=[MemoryType.CONVERSATION],
            agent_name=self.name,
            limit=10
        )
        
        collaboration_history = []
        for memory_entry, similarity in memories:
            if other_agent_name in memory_entry.content:
                collaboration_history.append({
                    "memory_id": memory_entry.id,
                    "content": memory_entry.content,
                    "timestamp": memory_entry.timestamp,
                    "similarity": similarity
                })
        
        return collaboration_history
    
    def _add_memory_context(self, memory_context: str):
        """Add memory context to agent's processing (implementation specific)."""
        # This would be implemented based on your specific agent architecture
        # For example, you might add it to the system message or context
        pass


class MemoryEnhancedMagenticOne:
    """Magentic-One team enhanced with persistent memory capabilities."""
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory_storage_path: str = "./magentic_memory"
    ):
        self.model_client = model_client
        self.memory_manager = PersistentMemoryManager(storage_path=memory_storage_path)
        self.knowledge_graph = KnowledgeGraph(storage_path=f"{memory_storage_path}/knowledge_graph")
        
        # Enhanced agents
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize memory-enhanced agents."""
        
        agent_configs = [
            {"name": "orchestrator", "description": "Coordinates multi-agent tasks"},
            {"name": "web_surfer", "description": "Browses and extracts web information"},
            {"name": "file_surfer", "description": "Navigates and reads local files"},
            {"name": "coder", "description": "Writes and executes code"},
            {"name": "computer_terminal", "description": "Executes system commands"}
        ]
        
        for config in agent_configs:
            agent = MemoryEnhancedAgent(
                name=config["name"],
                model_client=self.model_client,
                memory_manager=self.memory_manager,
                knowledge_graph=self.knowledge_graph,
                description=config["description"]
            )
            self.agents[config["name"]] = agent
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a task with memory enhancement."""
        
        # Check for similar past tasks
        similar_tasks = await self.memory_manager.find_similar_tasks(
            task_description=task_description,
            limit=3,
            similarity_threshold=0.8
        )
        
        execution_context = {
            "task": task_description,
            "similar_tasks": similar_tasks,
            "start_time": asyncio.get_event_loop().time()
        }
        
        try:
            # Execute task (simplified - would integrate with actual Magentic-One execution)
            result = await self._execute_with_orchestrator(task_description, execution_context)
            
            # Store successful execution
            if result.get("success"):
                await self._store_successful_execution(task_description, result)
            
            return result
            
        except Exception as e:
            # Store error for learning
            await self._store_execution_error(task_description, str(e), execution_context)
            raise
    
    async def _execute_with_orchestrator(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using the orchestrator (simplified implementation)."""
        
        # This would integrate with the actual Magentic-One orchestrator
        # For now, return a mock result
        return {
            "success": True,
            "result": f"Task completed: {task}",
            "agents_used": ["orchestrator", "web_surfer"],
            "execution_time": 10.5
        }
    
    async def _store_successful_execution(self, task: str, result: Dict[str, Any]):
        """Store successful task execution in memory."""
        
        solution_steps = [
            f"Used agents: {', '.join(result.get('agents_used', []))}",
            f"Execution time: {result.get('execution_time', 0):.1f}s",
            f"Result: {result.get('result', '')}"
        ]
        
        # Store in orchestrator's memory
        await self.agents["orchestrator"].store_task_success(
            task_description=task,
            solution_steps=solution_steps,
            success_rate=1.0
        )
    
    async def _store_execution_error(self, task: str, error: str, context: Dict[str, Any]):
        """Store execution error for learning."""
        
        await self.agents["orchestrator"].store_error_recovery(
            error_description=error,
            error_context=f"Task: {task}, Context: {context}",
            recovery_strategy="Review similar tasks and adjust approach"
        )
    
    async def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights from accumulated memory."""
        
        memory_stats = await self.memory_manager.get_memory_statistics()
        graph_stats = self.knowledge_graph.get_statistics()
        
        # Get most successful task patterns
        task_memories = await self.memory_manager.retrieve_memories(
            query_text="successful task completion",
            memory_types=[MemoryType.TASK_SOLUTION],
            limit=10
        )
        
        return {
            "memory_statistics": memory_stats,
            "knowledge_graph_statistics": graph_stats,
            "successful_patterns": [
                {
                    "content": memory.content[:200],
                    "confidence": memory.importance_score,
                    "agent": memory.agent_name
                }
                for memory, _ in task_memories
            ]
        }
    
    async def shutdown(self):
        """Shutdown the memory-enhanced system."""
        await self.memory_manager.shutdown()


# Example usage
async def example_usage():
    """Example of how to use the memory-enhanced Magentic-One system."""
    
    # Mock model client (replace with actual implementation)
    class MockModelClient:
        async def create(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "Mock response"}}]}
    
    model_client = MockModelClient()
    
    # Initialize memory-enhanced Magentic-One
    magentic_one = MemoryEnhancedMagenticOne(
        model_client=model_client,
        memory_storage_path="./example_memory"
    )
    
    # Execute tasks
    tasks = [
        "Research the latest developments in AI",
        "Create a Python script to analyze data",
        "Find information about climate change solutions"
    ]
    
    for task in tasks:
        print(f"Executing task: {task}")
        result = await magentic_one.execute_task(task)
        print(f"Result: {result}")
        print()
    
    # Get memory insights
    insights = await magentic_one.get_memory_insights()
    print("Memory Insights:")
    print(f"Total memories: {insights['memory_statistics']['vector_store_stats']['total_entries']}")
    print(f"Knowledge graph entities: {insights['knowledge_graph_statistics']['total_entities']}")
    
    # Shutdown
    await magentic_one.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())
