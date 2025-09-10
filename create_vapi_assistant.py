#!/usr/bin/env python3
"""
Create Vapi Assistant using API
Configures assistant for Magentic-One integration
"""

import requests
import json
import os

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
        return tool
    else:
        print(f"❌ Error creating tool: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def create_assistant(tool_id=None):
    """Create Vapi assistant configured for Magentic-One"""
    
    assistant_config = {
        "name": "Magentic-One General Agent",
        "firstMessage": "Hello! I'm your Magentic-One general agent. How can I help you today?",
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful general agent powered by Magentic-One. You can help with research, coding, web browsing, file operations, and complex multi-step tasks. When users ask for complex tasks, use the query_magentic_one function to process their request through the specialized agent system. Keep responses conversational and ask clarifying questions when needed."
                }
            ]
        },
        "voice": {
            "provider": "openai",
            "voiceId": "alloy"
        }
    }
    
    # Add tool if provided
    if tool_id:
        assistant_config["tools"] = [{"type": "function", "id": tool_id}]
    
    response = requests.post(
        f"{VAPI_BASE_URL}/assistant",
        headers=headers,
        json=assistant_config
    )
    
    if response.status_code == 201:
        assistant = response.json()
        print("✅ Assistant created successfully!")
        print(f"Assistant ID: {assistant['id']}")
        print(f"Name: {assistant['name']}")
        return assistant
    else:
        print(f"❌ Error creating assistant: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def create_phone_number(assistant_id):
    """Create a phone number and assign it to the assistant"""
    
    phone_config = {
        "assistantId": assistant_id,
        "server": {
            "url": WEBHOOK_URL
        }
    }
    
    response = requests.post(
        f"{VAPI_BASE_URL}/phone-number",
        headers=headers,
        json=phone_config
    )
    
    if response.status_code == 201:
        phone = response.json()
        print("✅ Phone number created successfully!")
        print(f"Phone Number: {phone['number']}")
        print(f"Assistant ID: {phone['assistantId']}")
        return phone
    else:
        print(f"❌ Error creating phone number: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def list_assistants():
    """List existing assistants"""
    response = requests.get(
        f"{VAPI_BASE_URL}/assistant",
        headers=headers
    )
    
    if response.status_code == 200:
        assistants = response.json()
        print(f"📋 Found {len(assistants)} assistants:")
        for assistant in assistants:
            print(f"  - {assistant['name']} (ID: {assistant['id']})")
        return assistants
    else:
        print(f"❌ Error listing assistants: {response.status_code}")
        return []

def main():
    print("🤖 Creating Vapi Assistant for Magentic-One...")
    print(f"Using API Key: {VAPI_API_KEY[:8]}...")
    print(f"Webhook URL: {WEBHOOK_URL}")
    print()
    
    # List existing assistants first
    print("📋 Checking existing assistants...")
    existing_assistants = list_assistants()
    print()
    
    # Create tool first
    print("🔧 Creating Magentic-One function tool...")
    tool = create_tool()
    
    if not tool:
        print("❌ Failed to create tool, cannot proceed")
        return
    
    print()
    
    # Create new assistant with tool
    print("🚀 Creating new assistant...")
    assistant = create_assistant(tool['id'])
    
    if assistant:
        print()
        print("📞 Creating phone number...")
        phone = create_phone_number(assistant['id'])
        
        if phone:
            print()
            print("🎉 Setup Complete!")
            print("=" * 50)
            print(f"Tool ID: {tool['id']}")
            print(f"Assistant ID: {assistant['id']}")
            print(f"Assistant Name: {assistant['name']}")
            print(f"Phone Number: {phone['number']}")
            print(f"Webhook URL: {WEBHOOK_URL}")
            print("=" * 50)
            print()
            print("Next steps:")
            print("1. Deploy the webhook server to handle Vapi events")
            print("2. Test the assistant by calling the phone number")
            print("3. Monitor logs for webhook events")
        else:
            print("⚠️  Assistant created but phone number setup failed")
            print(f"You can manually configure a phone number for assistant: {assistant['id']}")
    else:
        print("❌ Failed to create assistant")

if __name__ == "__main__":
    main()
