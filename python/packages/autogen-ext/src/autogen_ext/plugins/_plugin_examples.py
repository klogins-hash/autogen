"""
Example plugins demonstrating the AutoGen plugin architecture.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set
from autogen_agentchat.messages import TextMessage

from ._base_plugin import BasePlugin, PluginType, PluginCapability, PluginContext, plugin_hook
from ._agent_plugin import AgentPlugin, ToolPlugin, BehaviorPlugin, ToolDefinition
from ._plugin_manager import PluginManager


# Example usage and integration
async def example_plugin_usage():
    """Demonstrate how to use the plugin system with Magentic-One."""
    
    # Initialize plugin manager
    plugin_manager = PluginManager(
        plugin_directories=["./plugins", "./custom_plugins"],
        config_file="./plugin_config.json",
        auto_load=True
    )
    
    # Wait for auto-loading to complete
    await asyncio.sleep(1)
    
    # Get plugin statistics
    stats = plugin_manager.get_statistics()
    print(f"Loaded {stats['total_plugins']} plugins")
    print(f"Active plugins: {stats['plugins_by_status']['active']}")
    
    # Get plugins by capability
    logging_plugins = plugin_manager.get_plugins_by_capability(PluginCapability.LOGGING)
    tool_plugins = plugin_manager.get_plugins_by_capability(PluginCapability.TOOL_EXECUTION)
    
    print(f"Found {len(logging_plugins)} logging plugins")
    print(f"Found {len(tool_plugins)} tool plugins")
    
    # Execute hooks
    context = PluginContext(
        agent_name="test_agent",
        session_id="test_session",
        message_history=[],
        metadata={"test": True}
    )
    
    # Simulate message processing with hooks
    test_messages = [TextMessage(content="Hello, how can you help me?", source="user")]
    
    # Before processing hook
    await plugin_manager.execute_hook("before_message_processing", test_messages, context)
    
    # Simulate agent processing
    response = TextMessage(content="I can help you with various tasks!", source="assistant")
    
    # After processing hook
    await plugin_manager.execute_hook("after_message_processing", response, context)
    
    # Shutdown
    await plugin_manager.shutdown()


class EnhancedMagenticOneAgent:
    """Example of integrating plugins with a Magentic-One agent."""
    
    def __init__(self, name: str, plugin_manager: PluginManager):
        self.name = name
        self.plugin_manager = plugin_manager
        self.message_history: List[TextMessage] = []
    
    async def process_messages(self, messages: List[TextMessage]) -> TextMessage:
        """Process messages with plugin enhancement."""
        
        # Create context
        context = PluginContext(
            agent_name=self.name,
            session_id=f"session_{int(time.time())}",
            message_history=self.message_history,
            metadata={"agent_type": "magentic_one"}
        )
        
        # Pre-processing hooks
        processed_messages = messages
        for plugin in self.plugin_manager.get_plugins_by_capability(PluginCapability.MESSAGE_PROCESSING):
            if hasattr(plugin, 'pre_process_messages'):
                processed_messages = await plugin.pre_process_messages(processed_messages, context)
        
        # Execute before_message_processing hooks
        await self.plugin_manager.execute_hook("before_message_processing", processed_messages, context)
        
        # Core agent processing (simplified)
        response_content = f"Processed {len(processed_messages)} messages. "
        
        # Check for tool execution requests
        tool_plugins = self.plugin_manager.get_plugins_by_capability(PluginCapability.TOOL_EXECUTION)
        if tool_plugins and any("search" in msg.content.lower() for msg in processed_messages):
            # Execute web search tool
            for plugin in tool_plugins:
                if hasattr(plugin, 'execute_tool'):
                    try:
                        search_result = await plugin.execute_tool(
                            "web_search",
                            {"query": "latest AI developments", "max_results": 3},
                            context
                        )
                        response_content += f"Found search results: {search_result}"
                        break
                    except Exception:
                        continue
        
        response = TextMessage(content=response_content, source=self.name)
        
        # Post-processing hooks
        for plugin in self.plugin_manager.get_plugins_by_capability(PluginCapability.MESSAGE_PROCESSING):
            if hasattr(plugin, 'post_process_response'):
                response = await plugin.post_process_response(response, context)
        
        # Execute after_message_processing hooks
        await self.plugin_manager.execute_hook("after_message_processing", response, context)
        
        # Update message history
        self.message_history.extend(processed_messages)
        self.message_history.append(response)
        
        return response


# Custom plugin examples

class CustomAnalyticsPlugin(BehaviorPlugin):
    """Custom plugin for analytics and metrics collection."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "messages_processed": 0,
            "responses_generated": 0,
            "errors_encountered": 0,
            "average_response_time": 0.0
        }
        self.response_times = []
    
    @property
    def name(self) -> str:
        return "analytics_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Collects analytics and performance metrics"
    
    @property
    def capabilities(self) -> Set[PluginCapability]:
        return {PluginCapability.MONITORING, PluginCapability.LOGGING}
    
    def get_behavior_modifications(self) -> List:
        return []
    
    @plugin_hook("before_message_processing", priority=5)
    async def start_timing(self, messages: List[TextMessage], context: PluginContext) -> List[TextMessage]:
        """Start timing message processing."""
        context.set_shared_data("start_time", time.time())
        self.metrics["messages_processed"] += len(messages)
        return messages
    
    @plugin_hook("after_message_processing", priority=95)
    async def end_timing(self, response: TextMessage, context: PluginContext) -> TextMessage:
        """End timing and record metrics."""
        start_time = context.get_shared_data("start_time")
        if start_time:
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.metrics["average_response_time"] = sum(self.response_times) / len(self.response_times)
        
        self.metrics["responses_generated"] += 1
        return response
    
    @plugin_hook("on_error", priority=10)
    async def record_error(self, error: Exception, context: PluginContext) -> Optional[TextMessage]:
        """Record error metrics."""
        self.metrics["errors_encountered"] += 1
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()


class CustomSecurityPlugin(BehaviorPlugin):
    """Custom plugin for security and content filtering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocked_patterns = [
            "password", "secret", "api_key", "token",
            "credit_card", "ssn", "social_security"
        ]
        self.blocked_count = 0
    
    @property
    def name(self) -> str:
        return "security_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides security filtering and content validation"
    
    @property
    def capabilities(self) -> Set[PluginCapability]:
        return {PluginCapability.MESSAGE_PROCESSING, PluginCapability.AUTHENTICATION}
    
    def get_behavior_modifications(self) -> List:
        return []
    
    @plugin_hook("before_message_processing", priority=1)
    async def filter_sensitive_content(self, messages: List[TextMessage], context: PluginContext) -> List[TextMessage]:
        """Filter sensitive content from messages."""
        
        filtered_messages = []
        
        for message in messages:
            content = message.content.lower()
            
            # Check for blocked patterns
            contains_sensitive = any(pattern in content for pattern in self.blocked_patterns)
            
            if contains_sensitive:
                self.blocked_count += 1
                # Replace with filtered message
                filtered_message = TextMessage(
                    content="[FILTERED: Message contained sensitive information]",
                    source=message.source
                )
                filtered_messages.append(filtered_message)
            else:
                filtered_messages.append(message)
        
        return filtered_messages
    
    @plugin_hook("after_message_processing", priority=1)
    async def validate_response(self, response: TextMessage, context: PluginContext) -> TextMessage:
        """Validate response doesn't contain sensitive information."""
        
        content = response.content.lower()
        
        # Check for sensitive patterns in response
        contains_sensitive = any(pattern in content for pattern in self.blocked_patterns)
        
        if contains_sensitive:
            return TextMessage(
                content="I cannot provide information that might contain sensitive data.",
                source=response.source
            )
        
        return response
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            "blocked_messages": self.blocked_count,
            "blocked_patterns": self.blocked_patterns.copy()
        }


class CustomDatabaseToolPlugin(ToolPlugin):
    """Custom plugin providing database interaction tools."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connection_string = ""
        self.query_count = 0
    
    @property
    def name(self) -> str:
        return "database_tool"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides database interaction capabilities"
    
    async def _initialize(self) -> None:
        """Initialize database connection."""
        await super()._initialize()
        self.connection_string = self.get_setting("connection_string", "sqlite:///default.db")
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="execute_query",
                description="Execute a SQL query",
                parameters={
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "parameters": {"type": "array", "description": "Query parameters", "default": []}
                },
                function=self._execute_query,
                async_function=True,
                required_capabilities={"database_access"}
            ),
            ToolDefinition(
                name="get_table_schema",
                description="Get schema information for a table",
                parameters={
                    "table_name": {"type": "string", "description": "Name of the table"}
                },
                function=self._get_table_schema,
                async_function=True
            )
        ]
    
    async def _execute_query(self, query: str, parameters: List[Any] = None) -> Dict[str, Any]:
        """Execute a SQL query (mock implementation)."""
        
        self.query_count += 1
        
        # In a real implementation, you'd execute the actual query
        return {
            "success": True,
            "rows_affected": 5,
            "results": [
                {"id": 1, "name": "Sample Data 1"},
                {"id": 2, "name": "Sample Data 2"}
            ],
            "execution_time": 0.05
        }
    
    async def _get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema (mock implementation)."""
        
        return {
            "table_name": table_name,
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True},
                {"name": "name", "type": "VARCHAR(255)", "nullable": False},
                {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"}
            ]
        }
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        return {
            "total_queries": self.query_count,
            "connection_string": self.connection_string
        }


# Plugin configuration examples
def create_plugin_config():
    """Create example plugin configuration."""
    
    config = {
        "plugins": {
            "logging_plugin": {
                "enabled": True,
                "priority": 10,
                "settings": {
                    "log_file": "agent_interactions.log",
                    "log_level": "INFO"
                }
            },
            "caching_plugin": {
                "enabled": True,
                "priority": 20,
                "settings": {
                    "cache_size": 1000,
                    "cache_ttl": 3600
                }
            },
            "analytics_plugin": {
                "enabled": True,
                "priority": 5,
                "settings": {
                    "collect_detailed_metrics": True
                }
            },
            "security_plugin": {
                "enabled": True,
                "priority": 1,
                "settings": {
                    "strict_mode": True,
                    "additional_patterns": ["confidential", "private"]
                }
            },
            "database_tool": {
                "enabled": True,
                "priority": 50,
                "settings": {
                    "connection_string": "postgresql://user:pass@localhost/db",
                    "max_connections": 10
                }
            }
        },
        "global_settings": {
            "plugin_timeout": 30,
            "max_concurrent_plugins": 5,
            "error_handling": "continue"
        }
    }
    
    return config


# Integration example with Magentic-One
async def integrate_with_magentic_one():
    """Example of integrating plugins with Magentic-One."""
    
    # Create plugin manager
    plugin_manager = PluginManager(
        plugin_directories=["./magentic_plugins"],
        config_file="./magentic_plugin_config.json"
    )
    
    # Load custom plugins programmatically
    custom_plugins = [
        CustomAnalyticsPlugin(),
        CustomSecurityPlugin(),
        CustomDatabaseToolPlugin()
    ]
    
    for plugin in custom_plugins:
        # In a real implementation, you'd register these properly
        await plugin.initialize()
        await plugin.activate()
    
    # Create enhanced agent
    enhanced_agent = EnhancedMagenticOneAgent("orchestrator", plugin_manager)
    
    # Process some test messages
    test_messages = [
        TextMessage(content="Can you search for information about AI safety?", source="user"),
        TextMessage(content="What's my password for the database?", source="user"),  # Should be filtered
        TextMessage(content="Execute a query to get user statistics", source="user")
    ]
    
    for message in test_messages:
        print(f"Processing: {message.content}")
        response = await enhanced_agent.process_messages([message])
        print(f"Response: {response.content}")
        print()
    
    # Get plugin statistics
    for plugin in custom_plugins:
        if hasattr(plugin, 'get_metrics'):
            print(f"{plugin.name} metrics: {plugin.get_metrics()}")
        if hasattr(plugin, 'get_security_stats'):
            print(f"{plugin.name} security stats: {plugin.get_security_stats()}")
        if hasattr(plugin, 'get_query_stats'):
            print(f"{plugin.name} query stats: {plugin.get_query_stats()}")
    
    # Cleanup
    await plugin_manager.shutdown()


if __name__ == "__main__":
    # Run examples
    print("Running plugin system examples...")
    
    # Basic usage example
    asyncio.run(example_plugin_usage())
    
    print("\nRunning Magentic-One integration example...")
    
    # Magentic-One integration example
    asyncio.run(integrate_with_magentic_one())
    
    print("Examples completed!")
