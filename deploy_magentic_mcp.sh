#!/bin/bash

# Deploy Magentic-One MCP Server
# This script deploys a generalized MCP server for Magentic-One that can be used with any MCP client

set -e

SERVER="hitsdifferent"
DEPLOY_DIR="magentic-mcp"

echo "🚀 Deploying Magentic-One MCP Server to: $SERVER"

# Create deployment package
echo "📦 Creating MCP server deployment package..."
tar -czf magentic-mcp-deploy.tar.gz \
    enhanced_magentic_mcp_server.py \
    magentic_mcp_server.py \
    mcp_requirements.txt \
    test_mcp_server.py \
    start_mcp_server.sh

# Transfer files
echo "📤 Transferring files to $SERVER..."
scp magentic-mcp-deploy.tar.gz $SERVER:~/

# Setup on remote server
echo "🔧 Setting up MCP server on remote server..."
ssh $SERVER << 'EOF'
    # Create deployment directory
    mkdir -p ~/magentic-mcp
    cd ~/magentic-mcp
    
    # Extract files
    tar -xzf ~/magentic-mcp-deploy.tar.gz
    
    # Install dependencies
    echo "📦 Installing MCP dependencies..."
    python3 -m pip install --break-system-packages -r mcp_requirements.txt
    
    # Make server executable
    chmod +x magentic_mcp_server.py
    
    # Test enhanced MCP server
    echo "🧪 Testing enhanced MCP server..."
    python3 test_mcp_server.py
    
    # Make scripts executable
    chmod +x enhanced_magentic_mcp_server.py
    chmod +x start_mcp_server.sh
    
    echo "✅ MCP server setup complete!"
EOF

# Cleanup
rm -f magentic-mcp-deploy.tar.gz

echo ""
echo "🎉 Magentic-One MCP Server deployment completed!"
echo ""
echo "📋 Usage Instructions:"
echo "===================="
echo ""
echo "🔧 For Claude Desktop:"
echo "Add to ~/.config/claude/claude_desktop_config.json:"
echo '{'
echo '  "mcpServers": {'
echo '    "magentic-one": {'
echo '      "command": "ssh",'
echo '      "args": ["hitsdifferent", "cd ~/magentic-mcp && python3 magentic_mcp_server.py"]'
echo '    }'
echo '  }'
echo '}'
echo ""
echo "🌐 For LibreChat:"
echo "Add MCP server configuration to LibreChat environment"
echo ""
echo "🔌 For any MCP client:"
echo "Connect to: ssh hitsdifferent 'cd ~/magentic-mcp && python3 magentic_mcp_server.py'"
echo ""
echo "🛠️ Available Tools:"
echo "- query_magentic_one: Send queries to multi-agent system"
echo "- get_agent_status: Check agent system status"
echo "- execute_code: Run code through Coder agent"
echo "- browse_web: Browse web through WebSurfer agent"
echo "- analyze_files: Analyze files through FileSurfer agent"
echo ""
echo "📊 To check status:"
echo "  ssh $SERVER 'cd ~/magentic-mcp && python3 -c \"import docker; print(docker.from_env().containers.get(\\\"magentic-one-agent\\\").status)\"'"
