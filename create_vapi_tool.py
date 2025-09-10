#!/usr/bin/env python3
"""
Create Vapi Tool for Magentic-One Function
Creates the function tool first, then waits for confirmation
"""

import requests
import json

# Vapi API configuration
VAPI_API_KEY = "867ac81c-f57e-49ae-9003-25c88de12a15"
VAPI_BASE_URL = "https://api.vapi.ai"
WEBHOOK_URL = "https://206.189.203.174:8081/webhook"

headers = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json"
}

def create_tool():
    """Create the Magentic-One function tool"""
    
    tool_config = {
        "type": "function",
        "function": {
            "name": "query_magentic_one",
            "description": "Send complex queries to the Magentic-One general agent system for processing",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query or task to be processed by Magentic-One"
                    }
                },
                "required": ["query"]
            }
        },
        "server": {
            "url": WEBHOOK_URL
        }
    }
    
    response = requests.post(
        f"{VAPI_BASE_URL}/tool",
        headers=headers,
        json=tool_config
    )
    
    if response.status_code == 201:
        tool = response.json()
        print("✅ Tool created successfully!")
        print(f"Tool ID: {tool['id']}")
        print(f"Tool Name: {tool['function']['name']}")
        print(f"Tool Description: {tool['function']['description']}")
        print(f"Webhook URL: {tool['server']['url']}")
        return tool
    else:
        print(f"❌ Error creating tool: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def list_tools():
    """List existing tools"""
    response = requests.get(
        f"{VAPI_BASE_URL}/tool",
        headers=headers
    )
    
    if response.status_code == 200:
        tools = response.json()
        print(f"📋 Found {len(tools)} existing tools:")
        for tool in tools:
            if tool.get('function'):
                print(f"  - {tool['function']['name']} (ID: {tool['id']})")
            else:
                print(f"  - {tool.get('type', 'Unknown')} (ID: {tool['id']})")
        return tools
    else:
        print(f"❌ Error listing tools: {response.status_code}")
        return []

def main():
    print("🔧 Creating Vapi Tool for Magentic-One...")
    print(f"Using API Key: {VAPI_API_KEY[:8]}...")
    print(f"Webhook URL: {WEBHOOK_URL}")
    print()
    
    # List existing tools first
    print("📋 Checking existing tools...")
    existing_tools = list_tools()
    print()
    
    # Create new tool
    print("🚀 Creating Magentic-One function tool...")
    tool = create_tool()
    
    if tool:
        print()
        print("🎉 Tool Creation Complete!")
        print("=" * 50)
        print(f"Tool ID: {tool['id']}")
        print(f"Function Name: {tool['function']['name']}")
        print(f"Webhook URL: {WEBHOOK_URL}")
        print("=" * 50)
        print()
        print("✋ Ready to create assistant with this tool.")
        print("Please confirm before proceeding to assistant creation.")
    else:
        print("❌ Failed to create tool")

if __name__ == "__main__":
    main()
