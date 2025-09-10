#!/bin/bash

# Magentic-One + Vapi MCP Integration Deployment Script
# Usage: ./deploy_to_ssh.sh [ssh_host]

set -e

SSH_HOST=${1:-"hitsdifferent"}
REMOTE_DIR="/home/$(whoami)/magentic-one"

echo "🚀 Deploying Magentic-One + Vapi MCP Integration to SSH server: $SSH_HOST"

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf magentic-one-deploy.tar.gz \
    Dockerfile \
    docker-compose.yml \
    requirements.txt \
    magentic_one_setup.py \
    magentic_one_helper.py \
    vapi_mcp_node.js \
    vapi_mcp_integration.py \
    vapi_mcp_client.py \
    package.json \
    package-lock.json \
    mcp_requirements.txt \
    .env.voice \
    setup_instructions.md

# Transfer files to SSH server
echo "📤 Transferring files to $SSH_HOST..."
scp magentic-one-deploy.tar.gz $SSH_HOST:~/

# Deploy on remote server
echo "🔧 Setting up on remote server..."
ssh $SSH_HOST << 'EOF'
    # Create directory and extract files
    mkdir -p ~/magentic-one
    cd ~/magentic-one
    tar -xzf ~/magentic-one-deploy.tar.gz
    rm ~/magentic-one-deploy.tar.gz
    
    # Create workspace directory
    mkdir -p workspace
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "⚠️  Docker not found. Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        echo "✅ Docker installed. Please log out and back in for group changes to take effect."
    fi
    
    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null; then
        echo "⚠️  Docker Compose not found. Installing..."
        sudo apt-get update
        sudo apt-get install -y docker-compose-plugin
    fi
    
    # Install Node.js if not present (for MCP integration)
    if ! command -v node &> /dev/null; then
        echo "⚠️  Node.js not found. Installing..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    
    # Install npm dependencies for MCP integration
    echo "📦 Installing MCP dependencies..."
    npm install
    
    echo "🏗️  Building Magentic-One container..."
    docker compose build
    
    echo "✅ Deployment complete!"
    echo ""
    echo "🚀 To start the Magentic-One system:"
    echo "  cd ~/magentic-one"
    echo "  docker compose up -d"
    echo ""
    echo "🎤 To start the Vapi MCP integration:"
    echo "  node vapi_mcp_node.js"
    echo ""
    echo "🔧 To interact with Magentic-One directly:"
    echo "  docker compose exec magentic-one python magentic_one_setup.py"
    echo ""
    echo "📊 To view logs:"
    echo "  docker compose logs -f magentic-one"
EOF

# Clean up local deployment package
rm magentic-one-deploy.tar.gz

echo ""
echo "🎉 Deployment to $SSH_HOST completed successfully!"
echo ""
echo "🚀 Next steps to activate the full system:"
echo ""
echo "1. Start Magentic-One backend:"
echo "   ssh $SSH_HOST"
echo "   cd ~/magentic-one"
echo "   docker compose up -d"
echo ""
echo "2. Start Vapi MCP integration:"
echo "   node vapi_mcp_node.js"
echo ""
echo "3. Test the integration:"
echo "   - Type queries to send to Magentic-One"
echo "   - Use 'call <phone>' to make Vapi calls"
echo "   - Use 'quit' to exit"
echo ""
echo "🔑 Your Vapi Assistant ID: e820f3e6-7a17-432e-be14-5bf5cbf6e611"
echo "📞 Configure phone number in Vapi dashboard for full voice calling"
