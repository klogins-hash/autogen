#!/bin/bash

# LibreChat Deployment Script for SSH Server
# Usage: ./deploy_librechat.sh [ssh_host]

set -e

SSH_HOST=${1:-"hitsdifferent"}

echo "🚀 Deploying LibreChat to SSH server: $SSH_HOST"

# Create deployment package
echo "📦 Creating LibreChat deployment package..."
tar -czf librechat-deploy.tar.gz \
    librechat-docker-compose.yml \
    librechat.yaml \
    .env.production

# Transfer files to SSH server
echo "📤 Transferring files to $SSH_HOST..."
scp librechat-deploy.tar.gz $SSH_HOST:~/

# Deploy on remote server
echo "🔧 Setting up LibreChat on remote server..."
ssh $SSH_HOST << 'EOF'
    # Create LibreChat directory
    mkdir -p ~/librechat
    cd ~/librechat
    tar -xzf ~/librechat-deploy.tar.gz
    rm ~/librechat-deploy.tar.gz
    
    # Rename docker-compose file
    mv librechat-docker-compose.yml docker-compose.yml
    
    # Copy environment file
    cp .env.production .env
    
    # Create required directories
    mkdir -p logs images uploads data-node meili_data
    
    # Set permissions
    chmod 755 logs images uploads data-node meili_data
    
    echo "🏗️  Starting LibreChat services..."
    docker compose pull
    docker compose up -d
    
    echo "✅ LibreChat deployment complete!"
    echo ""
    echo "🌐 LibreChat will be available at:"
    echo "  http://206.189.203.174:3080"
    echo ""
    echo "📊 To view logs:"
    echo "  docker compose logs -f librechat"
    echo ""
    echo "🔧 To manage services:"
    echo "  docker compose stop    # Stop services"
    echo "  docker compose start   # Start services"
    echo "  docker compose restart # Restart services"
EOF

# Clean up local deployment package
rm librechat-deploy.tar.gz

echo ""
echo "🎉 LibreChat deployment to $SSH_HOST completed successfully!"
echo ""
echo "🌐 Access LibreChat at: http://206.189.203.174:3080"
echo "📝 Create an account and start chatting with your AI agents"
echo "🔗 LibreChat connects to your OpenAI API and Magentic-One system"
