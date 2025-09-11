"""
Specialized plugin types for AutoGen agents.
"""

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Union
from autogen_agentchat.messages import TextMessage

from ._base_plugin import BasePlugin, PluginType, PluginCapability, PluginContext, plugin_hook


@dataclass
class ToolDefinition:
    """Definition of a tool that can be used by agents."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    async_function: bool = False
    required_capabilities: Set[str] = field(default_factory=set)


@dataclass
class BehaviorModification:
    """Defines how a plugin modifies agent behavior."""
    target_method: str
    modification_type: str  # "before", "after", "replace", "wrap"
    priority: int = 100


class AgentPlugin(BasePlugin):
    """Base class for plugins that extend agent functionality."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.AGENT
    
    @abstractmethod
    async def process_message(
        self,
        message: TextMessage,
        context: PluginContext
    ) -> Optional[TextMessage]:
        """Process an incoming message."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[TextMessage],
        context: PluginContext
    ) -> Optional[TextMessage]:
        """Generate a response to messages."""
        pass
    
    async def pre_process_messages(
        self,
        messages: List[TextMessage],
        context: PluginContext
    ) -> List[TextMessage]:
        """Pre-process messages before agent handles them."""
        return messages
    
    async def post_process_response(
        self,
        response: TextMessage,
        context: PluginContext
    ) -> TextMessage:
        """Post-process agent response."""
        return response


class ToolPlugin(BasePlugin):
    """Plugin that provides tools for agents to use."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tools: Dict[str, ToolDefinition] = {}
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.TOOL
    
    @property
    def capabilities(self) -> Set[PluginCapability]:
        return {PluginCapability.TOOL_EXECUTION}
    
    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        """Get list of tools provided by this plugin."""
        pass
    
    async def _initialize(self) -> None:
        """Initialize tools."""
        tools = self.get_tools()
        for tool in tools:
            self._tools[tool.name] = tool
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: PluginContext
    ) -> Any:
        """Execute a tool with given parameters."""
        
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in plugin '{self.name}'")
        
        tool = self._tools[tool_name]
        
        # Validate parameters (simplified)
        # In a real implementation, you'd validate against the tool's parameter schema
        
        try:
            if tool.async_function:
                return await tool.function(**parameters)
            else:
                return tool.function(**parameters)
        except Exception as e:
            await self._handle_error(f"Tool execution failed: {e}")
            raise
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get definition for a specific tool."""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self._tools.keys())


class BehaviorPlugin(BasePlugin):
    """Plugin that modifies agent behavior through hooks and interceptors."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._behavior_modifications: List[BehaviorModification] = []
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.BEHAVIOR
    
    @property
    def capabilities(self) -> Set[PluginCapability]:
        return {PluginCapability.MESSAGE_PROCESSING, PluginCapability.CONTEXT_MODIFICATION}
    
    @abstractmethod
    def get_behavior_modifications(self) -> List[BehaviorModification]:
        """Get list of behavior modifications this plugin provides."""
        pass
    
    async def _initialize(self) -> None:
        """Initialize behavior modifications."""
        self._behavior_modifications = self.get_behavior_modifications()
    
    @plugin_hook("before_message_processing", priority=50)
    async def before_message_processing(
        self,
        messages: List[TextMessage],
        context: PluginContext
    ) -> List[TextMessage]:
        """Hook called before message processing."""
        return messages
    
    @plugin_hook("after_message_processing", priority=50)
    async def after_message_processing(
        self,
        response: TextMessage,
        context: PluginContext
    ) -> TextMessage:
        """Hook called after message processing."""
        return response
    
    @plugin_hook("on_error", priority=50)
    async def on_error(
        self,
        error: Exception,
        context: PluginContext
    ) -> Optional[TextMessage]:
        """Hook called when an error occurs."""
        return None


# Concrete plugin implementations

class LoggingPlugin(BehaviorPlugin):
    """Plugin that adds comprehensive logging to agent interactions."""
    
    @property
    def name(self) -> str:
        return "logging_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides comprehensive logging for agent interactions"
    
    @property
    def capabilities(self) -> Set[PluginCapability]:
        return {PluginCapability.LOGGING, PluginCapability.MONITORING}
    
    def get_behavior_modifications(self) -> List[BehaviorModification]:
        return [
            BehaviorModification("process_message", "before", 10),
            BehaviorModification("generate_response", "after", 10)
        ]
    
    async def _initialize(self) -> None:
        await super()._initialize()
        self.log_file = self.get_setting("log_file", "agent_interactions.log")
        self.log_level = self.get_setting("log_level", "INFO")
    
    @plugin_hook("before_message_processing", priority=10)
    async def log_incoming_message(
        self,
        messages: List[TextMessage],
        context: PluginContext
    ) -> List[TextMessage]:
        """Log incoming messages."""
        
        for message in messages:
            await self._log_message("INCOMING", message, context)
        
        return messages
    
    @plugin_hook("after_message_processing", priority=10)
    async def log_outgoing_response(
        self,
        response: TextMessage,
        context: PluginContext
    ) -> TextMessage:
        """Log outgoing responses."""
        
        await self._log_message("OUTGOING", response, context)
        return response
    
    async def _log_message(self, direction: str, message: TextMessage, context: PluginContext):
        """Log a message with context."""
        
        log_entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "direction": direction,
            "agent": context.agent_name,
            "session": context.session_id,
            "content": message.content[:200] + "..." if len(message.content) > 200 else message.content,
            "message_type": type(message).__name__
        }
        
        # In a real implementation, you'd write to a proper logging system
        print(f"[{self.name}] {log_entry}")


class CachingPlugin(BehaviorPlugin):
    """Plugin that adds response caching to reduce redundant processing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    @property
    def name(self) -> str:
        return "caching_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides response caching to reduce redundant processing"
    
    @property
    def capabilities(self) -> Set[PluginCapability]:
        return {PluginCapability.CACHING}
    
    def get_behavior_modifications(self) -> List[BehaviorModification]:
        return [
            BehaviorModification("generate_response", "wrap", 20)
        ]
    
    async def _initialize(self) -> None:
        await super()._initialize()
        self.cache_size = self.get_setting("cache_size", 1000)
        self.cache_ttl = self.get_setting("cache_ttl", 3600)  # 1 hour
    
    @plugin_hook("before_message_processing", priority=20)
    async def check_cache(
        self,
        messages: List[TextMessage],
        context: PluginContext
    ) -> List[TextMessage]:
        """Check if response is cached."""
        
        cache_key = self._generate_cache_key(messages, context)
        cached_response = self._get_cached_response(cache_key)
        
        if cached_response:
            self._cache_hits += 1
            context.set_shared_data("cached_response", cached_response)
        else:
            self._cache_misses += 1
        
        return messages
    
    @plugin_hook("after_message_processing", priority=20)
    async def cache_response(
        self,
        response: TextMessage,
        context: PluginContext
    ) -> TextMessage:
        """Cache the response."""
        
        # Only cache if not from cache
        if not context.get_shared_data("cached_response"):
            cache_key = self._generate_cache_key(context.message_history, context)
            self._cache_response(cache_key, response)
        
        return response
    
    def _generate_cache_key(self, messages: List[TextMessage], context: PluginContext) -> str:
        """Generate a cache key for messages."""
        
        content_hash = hash(tuple(msg.content for msg in messages[-3:]))  # Last 3 messages
        agent_hash = hash(context.agent_name)
        
        return f"{agent_hash}_{content_hash}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[TextMessage]:
        """Get cached response if available and not expired."""
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            current_time = asyncio.get_event_loop().time()
            
            if current_time - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["response"]
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: TextMessage) -> None:
        """Cache a response."""
        
        # Simple LRU eviction if cache is full
        if len(self._cache) >= self.cache_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        
        self._cache[cache_key] = {
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size
        }


class WebSearchToolPlugin(ToolPlugin):
    """Plugin that provides web search capabilities."""
    
    @property
    def name(self) -> str:
        return "web_search_tool"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides web search functionality for agents"
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="web_search",
                description="Search the web for information",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results", "default": 5}
                },
                function=self._web_search,
                async_function=True
            ),
            ToolDefinition(
                name="get_webpage_content",
                description="Get content from a specific webpage",
                parameters={
                    "url": {"type": "string", "description": "URL to fetch"},
                    "extract_text": {"type": "boolean", "description": "Extract text only", "default": True}
                },
                function=self._get_webpage_content,
                async_function=True
            )
        ]
    
    async def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform web search (mock implementation)."""
        
        # In a real implementation, you'd integrate with a search API
        mock_results = [
            {
                "title": f"Search result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a mock search result snippet for query '{query}'"
            }
            for i in range(min(max_results, 3))
        ]
        
        return mock_results
    
    async def _get_webpage_content(self, url: str, extract_text: bool = True) -> Dict[str, str]:
        """Get webpage content (mock implementation)."""
        
        # In a real implementation, you'd fetch and parse the webpage
        return {
            "url": url,
            "title": f"Mock Page Title for {url}",
            "content": f"Mock content from {url}. This would contain the actual webpage content.",
            "text_only": extract_text
        }


class CodeExecutionToolPlugin(ToolPlugin):
    """Plugin that provides code execution capabilities."""
    
    @property
    def name(self) -> str:
        return "code_execution_tool"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Provides safe code execution capabilities"
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="execute_python",
                description="Execute Python code safely",
                parameters={
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 30}
                },
                function=self._execute_python,
                async_function=True,
                required_capabilities={"code_execution"}
            ),
            ToolDefinition(
                name="validate_code",
                description="Validate code syntax without execution",
                parameters={
                    "code": {"type": "string", "description": "Code to validate"},
                    "language": {"type": "string", "description": "Programming language", "default": "python"}
                },
                function=self._validate_code,
                async_function=False
            )
        ]
    
    async def _execute_python(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code safely (mock implementation)."""
        
        # In a real implementation, you'd execute code in a sandboxed environment
        return {
            "success": True,
            "output": f"Mock execution output for code: {code[:50]}...",
            "execution_time": 0.1,
            "memory_used": "10MB"
        }
    
    def _validate_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code syntax."""
        
        if language.lower() == "python":
            try:
                compile(code, "<string>", "exec")
                return {"valid": True, "errors": []}
            except SyntaxError as e:
                return {"valid": False, "errors": [str(e)]}
        else:
            return {"valid": False, "errors": [f"Language '{language}' not supported"]}
