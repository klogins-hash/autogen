#!/bin/bash
# Enhanced Magentic-One MCP Server Startup Script

echo "🚀 Starting Enhanced Magentic-One MCP Server..."

# Set environment variables
export PYTHONPATH="/Users/franksimpson/CascadeProjects/autogen:/Users/franksimpson/CascadeProjects/autogen/python/packages/autogen-ext/src"

# Start the MCP server
/Users/franksimpson/CascadeProjects/autogen/venv-mcp/bin/python /Users/franksimpson/CascadeProjects/autogen/enhanced_magentic_mcp_server.py

echo "MCP Server stopped."
