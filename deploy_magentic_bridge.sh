#!/bin/bash

# Deploy Magentic-One API Bridge Script
# Usage: ./deploy_magentic_bridge.sh [ssh_host]

set -e

SSH_HOST=${1:-"hitsdifferent"}

echo "🚀 Deploying Magentic-One API Bridge to SSH server: $SSH_HOST"

# Create deployment package
echo "📦 Creating API bridge deployment package..."
tar -czf magentic-bridge-deploy.tar.gz \
    magentic_api_bridge.py \
    api_bridge_requirements.txt \
    librechat_magentic_config.yaml

# Transfer files to SSH server
echo "📤 Transferring files to $SSH_HOST..."
scp magentic-bridge-deploy.tar.gz $SSH_HOST:~/

# Deploy on remote server
echo "🔧 Setting up API bridge on remote server..."
ssh $SSH_HOST << 'EOF'
    # Create API bridge directory
    mkdir -p ~/magentic-bridge
    cd ~/magentic-bridge
    tar -xzf ~/magentic-bridge-deploy.tar.gz
    rm ~/magentic-bridge-deploy.tar.gz
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies..."
    python3 -m pip install --break-system-packages -r api_bridge_requirements.txt
    
    # Start the API bridge service
    echo "🚀 Starting Magentic-One API Bridge..."
    nohup python3 magentic_api_bridge.py > api_bridge.log 2>&1 &
    
    # Wait a moment for service to start
    sleep 3
    
    # Test the service
    echo "🧪 Testing API bridge..."
    curl -s http://localhost:8090/health || echo "Service starting..."
    
    echo "✅ API Bridge deployment complete!"
    echo ""
    echo "🌐 API Bridge running on: http://localhost:8090"
    echo "📊 Health check: http://localhost:8090/health"
    echo "📋 Models endpoint: http://localhost:8090/v1/models"
EOF

# Update LibreChat configuration
echo "🔧 Updating LibreChat configuration..."
ssh $SSH_HOST << 'EOF'
    cd ~/librechat
    
    # Backup original config
    cp librechat.yaml librechat.yaml.backup
    
    # Copy new config
    cp ~/magentic-bridge/librechat_magentic_config.yaml librechat.yaml
    
    # Restart LibreChat to pick up new config
    echo "🔄 Restarting LibreChat with Magentic-One integration..."
    docker compose restart librechat
    
    echo "✅ LibreChat updated with Magentic-One integration!"
EOF

# Clean up local deployment package
rm magentic-bridge-deploy.tar.gz

echo ""
echo "🎉 Magentic-One API Bridge deployment completed successfully!"
echo ""
echo "🌐 LibreChat now connects to Magentic-One at: http://206.189.203.174:3080"
echo "🤖 Select 'Magentic-One Multi-Agent' model in LibreChat to use your agents"
echo "⚡ API Bridge running on port 8090 (internal)"
echo ""
echo "📊 To check status:"
echo "  ssh $SSH_HOST 'curl http://localhost:8090/health'"
