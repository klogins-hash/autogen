#!/usr/bin/env python3
"""
Test script for the Enhanced Magentic-One MCP Server
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required imports work"""
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Resource, Tool, TextContent
        print("✅ MCP imports successful")
        return True
    except ImportError as e:
        print(f"❌ MCP import failed: {e}")
        return False

def test_enhanced_imports():
    """Test enhanced AutoGen component imports"""
    try:
        from python.packages.autogen_ext.src.autogen_ext.security import SecurityIntegration
        print("✅ Security integration import successful")
    except ImportError:
        print("⚠️  Security integration not available (optional)")
    
    try:
        from python.packages.autogen_ext.src.autogen_ext.dashboard import DashboardIntegration
        print("✅ Dashboard integration import successful")
    except ImportError:
        print("⚠️  Dashboard integration not available (optional)")
    
    try:
        from python.packages.autogen_ext.src.autogen_ext.plugins import PluginManager
        print("✅ Plugin manager import successful")
    except ImportError:
        print("⚠️  Plugin manager not available (optional)")

def test_server_creation():
    """Test server creation"""
    try:
        from enhanced_magentic_mcp_server import EnhancedMagenticMCPServer
        server = EnhancedMagenticMCPServer()
        print("✅ Enhanced MCP Server created successfully")
        return True
    except Exception as e:
        print(f"❌ Server creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced Magentic-One MCP Server...")
    print("=" * 50)
    
    # Test basic MCP imports
    mcp_ok = test_imports()
    
    # Test enhanced component imports
    test_enhanced_imports()
    
    # Test server creation
    if mcp_ok:
        server_ok = test_server_creation()
        if server_ok:
            print("\n🎉 All tests passed! MCP server is ready.")
        else:
            print("\n❌ Server creation failed.")
    else:
        print("\n❌ Basic MCP imports failed.")
