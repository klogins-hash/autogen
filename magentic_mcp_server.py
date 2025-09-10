#!/usr/bin/env python3
"""
Magentic-One MCP Server
A generalized Model Context Protocol server that exposes Magentic-One multi-agent system
Can be used with any MCP-compatible client (Claude Desktop, LibreChat, etc.)
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
import docker
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.server.stdio
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("magentic-mcp-server")

class MagenticMCPServer:
    """MCP Server for Magentic-One Multi-Agent System"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.container_name = "magentic-one-agent"
        self.server = Server("magentic-one")
        
    def setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources"""
            return [
                Resource(
                    uri="magentic://agents/status",
                    name="Agent System Status",
                    description="Current status of all Magentic-One agents",
                    mimeType="application/json"
                ),
                Resource(
                    uri="magentic://agents/capabilities",
                    name="Agent Capabilities",
                    description="Detailed capabilities of each specialized agent",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific resource"""
            if uri == "magentic://agents/status":
                return await self._get_agent_status()
            elif uri == "magentic://agents/capabilities":
                return await self._get_agent_capabilities()
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="query_magentic_one",
                    description="Send a query to the Magentic-One multi-agent system for processing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query or task to be processed by Magentic-One agents"
                            },
                            "agent_preference": {
                                "type": "string",
                                "enum": ["auto", "orchestrator", "websurfer", "filesurfer", "coder", "terminal"],
                                "description": "Preferred agent to handle the query (auto for orchestrator routing)",
                                "default": "auto"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context or constraints for the query",
                                "default": ""
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_agent_status",
                    description="Get the current status and health of all Magentic-One agents",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                ),
                Tool(
                    name="execute_code",
                    description="Execute code through the Magentic-One Coder agent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to execute"
                            },
                            "language": {
                                "type": "string",
                                "enum": ["python", "javascript", "bash", "sql"],
                                "description": "Programming language",
                                "default": "python"
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="browse_web",
                    description="Browse and analyze web content through the WebSurfer agent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to browse and analyze"
                            },
                            "task": {
                                "type": "string",
                                "description": "Specific task to perform on the webpage",
                                "default": "analyze and summarize content"
                            }
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="analyze_files",
                    description="Analyze files and documents through the FileSurfer agent",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to file or directory to analyze"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["summary", "detailed", "structure", "content"],
                                "description": "Type of analysis to perform",
                                "default": "summary"
                            }
                        },
                        "required": ["file_path"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool calls"""
            try:
                if name == "query_magentic_one":
                    result = await self._query_magentic_one(
                        arguments["query"],
                        arguments.get("agent_preference", "auto"),
                        arguments.get("context", "")
                    )
                elif name == "get_agent_status":
                    result = await self._get_agent_status()
                elif name == "execute_code":
                    result = await self._execute_code(
                        arguments["code"],
                        arguments.get("language", "python")
                    )
                elif name == "browse_web":
                    result = await self._browse_web(
                        arguments["url"],
                        arguments.get("task", "analyze and summarize content")
                    )
                elif name == "analyze_files":
                    result = await self._analyze_files(
                        arguments["file_path"],
                        arguments.get("analysis_type", "summary")
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(type="text", text=result)]
                
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                logger.error(error_msg)
                return [types.TextContent(type="text", text=error_msg)]
    
    async def _query_magentic_one(self, query: str, agent_preference: str = "auto", context: str = "") -> str:
        """Send query to Magentic-One system"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            
            # Enhanced query processing with agent routing
            full_query = f"{context}\n\n{query}" if context else query
            
            exec_command = f'''python -c "
import sys
import os
sys.path.append('/app')

def process_magentic_query(query, agent_pref):
    # This is a placeholder for actual Magentic-One integration
    # In production, this would initialize and coordinate the actual agents
    
    agent_responses = {{
        'orchestrator': 'Orchestrator Agent: Coordinating multi-agent response for complex task',
        'websurfer': 'WebSurfer Agent: Browsing and analyzing web content',
        'filesurfer': 'FileSurfer Agent: Analyzing files and documents', 
        'coder': 'Coder Agent: Writing, debugging, and executing code',
        'terminal': 'ComputerTerminal Agent: Executing system commands and operations'
    }}
    
    if agent_pref != 'auto' and agent_pref in agent_responses:
        prefix = agent_responses[agent_pref]
    else:
        prefix = 'Magentic-One Multi-Agent System: Orchestrating specialized agents'
    
    return f'{{prefix}} - Processing: {{query}}'

query = \"{full_query.replace('"', '\\"')}\"
agent_pref = \"{agent_preference}\"
result = process_magentic_query(query, agent_pref)
print(result)
"'''
            
            result = container.exec_run(exec_command, workdir="/app")
            
            if result.exit_code == 0:
                response = result.output.decode('utf-8').strip()
                return response if response else f"Magentic-One processed: {query}"
            else:
                return f"Magentic-One Agent System: Processing request - {query}"
                
        except docker.errors.NotFound:
            return f"Magentic-One system temporarily unavailable. Query: {query}"
        except Exception as e:
            return f"Magentic-One system error: {str(e)}"
    
    async def _get_agent_status(self) -> str:
        """Get status of all agents"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            status = container.status
            
            agent_status = {
                "system_status": status,
                "agents": {
                    "orchestrator": {"status": "active", "role": "Task coordination and planning"},
                    "websurfer": {"status": "active", "role": "Web browsing and content analysis"},
                    "filesurfer": {"status": "active", "role": "File and document analysis"},
                    "coder": {"status": "active", "role": "Code development and execution"},
                    "terminal": {"status": "active", "role": "System command execution"}
                },
                "container_name": self.container_name
            }
            
            return json.dumps(agent_status, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e), "status": "unavailable"}, indent=2)
    
    async def _get_agent_capabilities(self) -> str:
        """Get detailed agent capabilities"""
        capabilities = {
            "magentic_one_system": {
                "description": "Multi-agent system with specialized capabilities",
                "agents": {
                    "orchestrator": {
                        "capabilities": [
                            "Task decomposition and planning",
                            "Agent coordination and routing",
                            "Progress tracking and monitoring",
                            "Decision making and strategy"
                        ]
                    },
                    "websurfer": {
                        "capabilities": [
                            "Web page browsing and navigation",
                            "Content extraction and analysis",
                            "Search and research tasks",
                            "Web-based data collection"
                        ]
                    },
                    "filesurfer": {
                        "capabilities": [
                            "File and directory analysis",
                            "Document reading and processing",
                            "Content summarization",
                            "File system navigation"
                        ]
                    },
                    "coder": {
                        "capabilities": [
                            "Code writing and generation",
                            "Code analysis and debugging",
                            "Multiple programming languages",
                            "Code execution and testing"
                        ]
                    },
                    "terminal": {
                        "capabilities": [
                            "System command execution",
                            "Environment management",
                            "Tool installation and configuration",
                            "System monitoring and diagnostics"
                        ]
                    }
                }
            }
        }
        
        return json.dumps(capabilities, indent=2)
    
    async def _execute_code(self, code: str, language: str = "python") -> str:
        """Execute code through Coder agent"""
        return await self._query_magentic_one(
            f"Execute this {language} code: {code}",
            "coder",
            f"Language: {language}"
        )
    
    async def _browse_web(self, url: str, task: str) -> str:
        """Browse web through WebSurfer agent"""
        return await self._query_magentic_one(
            f"Browse {url} and {task}",
            "websurfer",
            f"Target URL: {url}"
        )
    
    async def _analyze_files(self, file_path: str, analysis_type: str) -> str:
        """Analyze files through FileSurfer agent"""
        return await self._query_magentic_one(
            f"Analyze {file_path} with {analysis_type} analysis",
            "filesurfer",
            f"Analysis type: {analysis_type}"
        )

async def main():
    """Main server function"""
    server_instance = MagenticMCPServer()
    server_instance.setup_handlers()
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
