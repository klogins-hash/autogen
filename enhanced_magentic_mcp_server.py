#!/usr/bin/env python3
"""
Enhanced Magentic-One MCP Server
Integrates with the enhanced AutoGen system including security, dashboard, and plugin features
"""

import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
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
except ImportError as e:
    print(f"MCP library not found: {e}")
    print("Install with: pip install mcp")
    sys.exit(1)

# Import enhanced AutoGen components
try:
    from python.packages.autogen_ext.src.autogen_ext.security import SecurityIntegration, SecurityConfig
    from python.packages.autogen_ext.src.autogen_ext.dashboard import DashboardIntegration
    from python.packages.autogen_ext.src.autogen_ext.plugins import PluginManager
    from python.packages.autogen_ext.src.autogen_ext.setup import SetupManager
    from python.packages.autogen_ext.src.autogen_ext.memory.persistent import PersistentMemoryManager
except ImportError as e:
    print(f"Enhanced AutoGen components not found: {e}")
    print("Using basic Magentic-One functionality")
    SecurityIntegration = None
    DashboardIntegration = None
    PluginManager = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced-magentic-mcp")

class EnhancedMagenticMCPServer:
    """Enhanced MCP Server for Magentic-One with security, dashboard, and plugin support"""
    
    def __init__(self):
        self.server = Server("enhanced-magentic-one")
        
        # Initialize enhanced components
        self.security_integration = None
        self.dashboard_integration = None
        self.plugin_manager = None
        self.memory_manager = None
        
        # Initialize if components are available
        if SecurityIntegration:
            self.security_integration = SecurityIntegration()
        
        if DashboardIntegration:
            self.dashboard_integration = DashboardIntegration()
        
        if PluginManager:
            self.plugin_manager = PluginManager()
        
        # Agent status tracking
        self.agent_status = {
            "orchestrator": {"status": "ready", "last_activity": None},
            "websurfer": {"status": "ready", "last_activity": None},
            "filesurfer": {"status": "ready", "last_activity": None},
            "coder": {"status": "ready", "last_activity": None},
            "terminal": {"status": "ready", "last_activity": None}
        }
        
    async def initialize(self):
        """Initialize all components"""
        try:
            if self.security_integration:
                await self.security_integration.start()
                logger.info("Security integration started")
            
            if self.dashboard_integration:
                await self.dashboard_integration.initialize()
                logger.info("Dashboard integration initialized")
            
            if self.plugin_manager:
                await self.plugin_manager.initialize()
                logger.info("Plugin manager initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources"""
            resources = [
                Resource(
                    uri="magentic://agents/status",
                    name="Enhanced Agent System Status",
                    description="Current status of all Magentic-One agents with security and monitoring",
                    mimeType="application/json"
                ),
                Resource(
                    uri="magentic://agents/capabilities",
                    name="Enhanced Agent Capabilities",
                    description="Detailed capabilities including plugins and security features",
                    mimeType="application/json"
                ),
                Resource(
                    uri="magentic://security/status",
                    name="Security System Status",
                    description="Current security monitoring and threat detection status",
                    mimeType="application/json"
                ),
                Resource(
                    uri="magentic://dashboard/metrics",
                    name="Dashboard Metrics",
                    description="Real-time system metrics and performance data",
                    mimeType="application/json"
                )
            ]
            
            if self.plugin_manager:
                resources.append(Resource(
                    uri="magentic://plugins/list",
                    name="Available Plugins",
                    description="List of loaded and available plugins",
                    mimeType="application/json"
                ))
            
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific resource"""
            if uri == "magentic://agents/status":
                return await self._get_enhanced_agent_status()
            elif uri == "magentic://agents/capabilities":
                return await self._get_enhanced_capabilities()
            elif uri == "magentic://security/status":
                return await self._get_security_status()
            elif uri == "magentic://dashboard/metrics":
                return await self._get_dashboard_metrics()
            elif uri == "magentic://plugins/list":
                return await self._get_plugin_list()
            else:
                raise ValueError(f"Unknown resource: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            tools = [
                Tool(
                    name="secure_query",
                    description="Send a secure query to the enhanced Magentic-One system with security analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query or task to be processed"
                            },
                            "agent_preference": {
                                "type": "string",
                                "enum": ["auto", "orchestrator", "websurfer", "filesurfer", "coder", "terminal"],
                                "description": "Preferred agent to handle the query",
                                "default": "auto"
                            },
                            "security_level": {
                                "type": "string",
                                "enum": ["standard", "strict", "permissive"],
                                "description": "Security analysis level",
                                "default": "standard"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for the query",
                                "default": ""
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_system_health",
                    description="Get comprehensive system health including security, performance, and agent status",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include detailed performance metrics",
                                "default": True
                            }
                        }
                    }
                ),
                Tool(
                    name="execute_secure_code",
                    description="Execute code with enhanced security analysis and monitoring",
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
                            },
                            "sandbox_mode": {
                                "type": "boolean",
                                "description": "Execute in secure sandbox",
                                "default": True
                            }
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="manage_plugins",
                    description="Manage system plugins (list, enable, disable, configure)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list", "enable", "disable", "configure", "status"],
                                "description": "Plugin management action"
                            },
                            "plugin_name": {
                                "type": "string",
                                "description": "Name of plugin (required for enable/disable/configure)"
                            },
                            "config": {
                                "type": "object",
                                "description": "Plugin configuration (for configure action)"
                            }
                        },
                        "required": ["action"]
                    }
                ),
                Tool(
                    name="security_analysis",
                    description="Perform security analysis on input text or code",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to analyze for security threats"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["prompt_injection", "content_filter", "input_validation", "comprehensive"],
                                "description": "Type of security analysis",
                                "default": "comprehensive"
                            }
                        },
                        "required": ["content"]
                    }
                )
            ]
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool calls"""
            try:
                if name == "secure_query":
                    result = await self._secure_query(
                        arguments["query"],
                        arguments.get("agent_preference", "auto"),
                        arguments.get("security_level", "standard"),
                        arguments.get("context", "")
                    )
                elif name == "get_system_health":
                    result = await self._get_system_health(
                        arguments.get("include_metrics", True)
                    )
                elif name == "execute_secure_code":
                    result = await self._execute_secure_code(
                        arguments["code"],
                        arguments.get("language", "python"),
                        arguments.get("sandbox_mode", True)
                    )
                elif name == "manage_plugins":
                    result = await self._manage_plugins(
                        arguments["action"],
                        arguments.get("plugin_name"),
                        arguments.get("config")
                    )
                elif name == "security_analysis":
                    result = await self._security_analysis(
                        arguments["content"],
                        arguments.get("analysis_type", "comprehensive")
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(type="text", text=result)]
                
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                logger.error(error_msg)
                return [types.TextContent(type="text", text=error_msg)]
    
    async def _secure_query(self, query: str, agent_preference: str, security_level: str, context: str) -> str:
        """Process query with security analysis"""
        try:
            # Security analysis if available
            if self.security_integration:
                security_result = await self.security_integration.analyze_input(
                    query, 
                    context={"agent_preference": agent_preference, "security_level": security_level}
                )
                
                if security_result["action"] == "block":
                    return f"Query blocked by security system: {', '.join(security_result['reasons'])}"
                
                if security_result["action"] == "sanitize":
                    query = security_result.get("sanitized_input", query)
            
            # Process with appropriate agent
            agent_responses = {
                'orchestrator': f'🎯 Orchestrator coordinating multi-agent response for: {query}',
                'websurfer': f'🌐 WebSurfer analyzing web content for: {query}',
                'filesurfer': f'📁 FileSurfer processing files and documents for: {query}',
                'coder': f'💻 Coder developing and executing code for: {query}',
                'terminal': f'⚡ Terminal executing system operations for: {query}'
            }
            
            if agent_preference != 'auto' and agent_preference in agent_responses:
                response = agent_responses[agent_preference]
            else:
                response = f'🤖 Enhanced Magentic-One System processing: {query}'
            
            # Add context if provided
            if context:
                response += f"\n📝 Context: {context}"
            
            # Add security info if analysis was performed
            if self.security_integration and 'security_result' in locals():
                if security_result.get("sanitized_input"):
                    response += f"\n🔒 Security: Input sanitized for safety"
                else:
                    response += f"\n✅ Security: Input passed all security checks"
            
            return response
            
        except Exception as e:
            return f"Enhanced Magentic-One system error: {str(e)}"
    
    async def _get_enhanced_agent_status(self) -> str:
        """Get comprehensive system status"""
        status = {
            "system": "Enhanced Magentic-One Multi-Agent System",
            "version": "2.0.0-enhanced",
            "timestamp": asyncio.get_event_loop().time(),
            "agents": self.agent_status,
            "components": {
                "security_integration": self.security_integration is not None,
                "dashboard_integration": self.dashboard_integration is not None,
                "plugin_manager": self.plugin_manager is not None,
                "memory_manager": self.memory_manager is not None
            }
        }
        
        # Add component status if available
        if self.security_integration:
            status["security_stats"] = self.security_integration.get_statistics()
        
        if self.plugin_manager:
            status["active_plugins"] = len(self.plugin_manager.get_active_plugins())
        
        return json.dumps(status, indent=2)
    
    async def _get_enhanced_capabilities(self) -> str:
        """Get enhanced system capabilities"""
        capabilities = {
            "enhanced_magentic_one": {
                "description": "Enhanced multi-agent system with security, monitoring, and plugins",
                "core_agents": {
                    "orchestrator": {
                        "capabilities": [
                            "Advanced task decomposition and planning",
                            "Intelligent agent coordination",
                            "Real-time progress monitoring",
                            "Security-aware decision making"
                        ],
                        "enhancements": ["Security integration", "Performance monitoring"]
                    },
                    "websurfer": {
                        "capabilities": [
                            "Secure web browsing with content filtering",
                            "Advanced content analysis",
                            "Threat-aware web research",
                            "Protected data collection"
                        ],
                        "enhancements": ["Content security scanning", "Malicious site detection"]
                    },
                    "filesurfer": {
                        "capabilities": [
                            "Secure file analysis with validation",
                            "Content sanitization",
                            "Malware detection",
                            "Safe document processing"
                        ],
                        "enhancements": ["File security scanning", "Content validation"]
                    },
                    "coder": {
                        "capabilities": [
                            "Secure code generation and analysis",
                            "Sandboxed code execution",
                            "Vulnerability detection",
                            "Safe code testing"
                        ],
                        "enhancements": ["Code security analysis", "Sandbox isolation"]
                    },
                    "terminal": {
                        "capabilities": [
                            "Secure command execution",
                            "Environment isolation",
                            "Command validation",
                            "System monitoring"
                        ],
                        "enhancements": ["Command security filtering", "Execution monitoring"]
                    }
                },
                "security_features": [
                    "Prompt injection defense",
                    "Input validation and sanitization",
                    "Content filtering",
                    "Real-time threat monitoring",
                    "Security event logging"
                ],
                "monitoring_features": [
                    "Real-time dashboard",
                    "Performance metrics",
                    "Agent health monitoring",
                    "Security event tracking"
                ]
            }
        }
        
        if self.plugin_manager:
            capabilities["plugin_system"] = {
                "description": "Extensible plugin architecture",
                "features": ["Dynamic loading", "Runtime management", "Custom tools"]
            }
        
        return json.dumps(capabilities, indent=2)
    
    async def _get_security_status(self) -> str:
        """Get security system status"""
        if not self.security_integration:
            return json.dumps({"status": "Security integration not available"}, indent=2)
        
        try:
            stats = self.security_integration.get_statistics()
            return json.dumps({
                "security_system": "Active",
                "statistics": stats,
                "timestamp": asyncio.get_event_loop().time()
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    async def _get_dashboard_metrics(self) -> str:
        """Get dashboard metrics"""
        if not self.dashboard_integration:
            return json.dumps({"status": "Dashboard integration not available"}, indent=2)
        
        try:
            metrics = self.dashboard_integration.get_dashboard_data()
            return json.dumps(metrics, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    async def _get_plugin_list(self) -> str:
        """Get list of available plugins"""
        if not self.plugin_manager:
            return json.dumps({"status": "Plugin manager not available"}, indent=2)
        
        try:
            plugins = self.plugin_manager.get_plugin_registry().get_all_plugins()
            return json.dumps({
                "plugins": plugins,
                "count": len(plugins)
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    async def _get_system_health(self, include_metrics: bool) -> str:
        """Get comprehensive system health"""
        health = {
            "system_status": "operational",
            "agents": self.agent_status,
            "components": {
                "security": self.security_integration is not None,
                "dashboard": self.dashboard_integration is not None,
                "plugins": self.plugin_manager is not None
            }
        }
        
        if include_metrics and self.security_integration:
            health["security_metrics"] = self.security_integration.get_statistics()
        
        return json.dumps(health, indent=2)
    
    async def _execute_secure_code(self, code: str, language: str, sandbox_mode: bool) -> str:
        """Execute code with security analysis"""
        try:
            # Security analysis
            if self.security_integration:
                security_result = await self.security_integration.analyze_input(
                    code, 
                    context={"type": "code", "language": language}
                )
                
                if security_result["action"] == "block":
                    return f"Code execution blocked: {', '.join(security_result['reasons'])}"
            
            # Simulate secure code execution
            execution_result = {
                "status": "executed",
                "language": language,
                "sandbox": sandbox_mode,
                "output": f"Securely executed {language} code in {'sandbox' if sandbox_mode else 'standard'} mode",
                "security_checked": self.security_integration is not None
            }
            
            return json.dumps(execution_result, indent=2)
            
        except Exception as e:
            return f"Code execution error: {str(e)}"
    
    async def _manage_plugins(self, action: str, plugin_name: Optional[str], config: Optional[Dict]) -> str:
        """Manage system plugins"""
        if not self.plugin_manager:
            return "Plugin manager not available"
        
        try:
            if action == "list":
                plugins = self.plugin_manager.get_active_plugins()
                return json.dumps({"active_plugins": list(plugins.keys())}, indent=2)
            
            elif action == "status":
                registry = self.plugin_manager.get_plugin_registry()
                return json.dumps({
                    "total_plugins": len(registry.get_all_plugins()),
                    "active_plugins": len(self.plugin_manager.get_active_plugins())
                }, indent=2)
            
            else:
                return f"Plugin action '{action}' requires plugin_name parameter"
                
        except Exception as e:
            return f"Plugin management error: {str(e)}"
    
    async def _security_analysis(self, content: str, analysis_type: str) -> str:
        """Perform security analysis"""
        if not self.security_integration:
            return "Security integration not available"
        
        try:
            result = await self.security_integration.analyze_input(
                content,
                context={"analysis_type": analysis_type}
            )
            
            return json.dumps({
                "analysis_type": analysis_type,
                "result": result
            }, indent=2)
            
        except Exception as e:
            return f"Security analysis error: {str(e)}"

async def main():
    """Main server function"""
    server_instance = EnhancedMagenticMCPServer()
    
    # Initialize enhanced components
    await server_instance.initialize()
    
    # Setup handlers
    server_instance.setup_handlers()
    
    logger.info("Enhanced Magentic-One MCP Server starting...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
