#!/usr/bin/env python3
"""
Fix Vapi Assistant Tool Connection - Updated API Format
"""

import requests
import json

# Configuration
VAPI_API_KEY = "867ac81c-f57e-49ae-9003-25c88de12a15"
ASSISTANT_ID = "e820f3e6-7a17-432e-be14-5bf5cbf6e611"
TOOL_ID = "bd815daa-5d73-4141-99e4-76374575f4e1"

def update_assistant_with_tool():
    """Update assistant to use the proper tool connection"""
    
    url = f"https://api.vapi.ai/assistant/{ASSISTANT_ID}"
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Correct API format for assistant update
    assistant_config = {
        "name": "Magentic-One General Agent",
        "firstMessage": "Hello! I'm your Magentic-One General Agent. I can help you with research, coding, analysis, and complex tasks using my specialized agent team. What can I help you with today?",
        
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "maxTokens": 1000
        },
        
        "systemMessage": """You are the Magentic-One General Agent, a sophisticated AI assistant powered by a multi-agent system.

## Identity & Capabilities
You have access to specialized agents: Web Research, File Analysis, Code Development, System Operations, and Task Orchestration.

## Communication Style
- **Concise & Clear**: Brief responses optimized for voice
- **Professional**: Expert but approachable
- **Direct**: Answer questions directly before context

## Tool Usage
For complex queries requiring research, analysis, coding, or multi-step tasks, use the query_magentic_one function to leverage the full multi-agent system.

## Response Guidelines
- Keep responses under 100 words when possible
- Be confident in your capabilities
- Handle errors gracefully with alternatives""",
        
        # Voice configuration
        "voice": {
            "provider": "openai",
            "voiceId": "alloy",
            "speed": 1.1
        },
        
        # Tool connection - Use proper format
        "toolIds": [TOOL_ID],
        
        # Speech timing
        "startSpeakingPlan": {
            "waitSeconds": 0.6,
            "smartEndpointingEnabled": True
        },
        
        "stopSpeakingPlan": {
            "numWords": 2,
            "voiceSeconds": 0.3,
            "backoffSeconds": 0.8
        },
        
        # Transcription
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en"
        }
    }
    
    try:
        print(f"🔧 Updating assistant {ASSISTANT_ID} to use tool {TOOL_ID}...")
        
        response = requests.patch(url, headers=headers, json=assistant_config)
        
        if response.status_code == 200:
            assistant_data = response.json()
            print("✅ Assistant updated successfully!")
            print(f"📋 Assistant: {assistant_data.get('name')}")
            print(f"🔧 Tool IDs: {assistant_data.get('toolIds', [])}")
            return assistant_data
        else:
            print(f"❌ Error updating assistant: {response.status_code}")
            print(f"Response: {response.text}")
            
            # Try alternative approach - just update toolIds
            simple_config = {"toolIds": [TOOL_ID]}
            response2 = requests.patch(url, headers=headers, json=simple_config)
            
            if response2.status_code == 200:
                print("✅ Tool connection updated with simplified approach!")
                return response2.json()
            else:
                print(f"❌ Alternative approach also failed: {response2.status_code}")
                print(f"Response: {response2.text}")
            
            return None
            
    except Exception as e:
        print(f"❌ Exception updating assistant: {str(e)}")
        return None

def verify_tool_connection():
    """Verify the tool is properly connected"""
    
    url = f"https://api.vapi.ai/assistant/{ASSISTANT_ID}"
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"🔍 Verifying assistant configuration...")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            assistant_data = response.json()
            tool_ids = assistant_data.get('toolIds', [])
            
            print(f"📋 Assistant: {assistant_data.get('name')}")
            print(f"🔧 Connected Tools: {len(tool_ids)}")
            
            if TOOL_ID in tool_ids:
                print(f"✅ Tool {TOOL_ID} is properly connected!")
                return True
            else:
                print(f"❌ Tool {TOOL_ID} is NOT connected")
                print(f"Current tool IDs: {tool_ids}")
                return False
        else:
            print(f"❌ Error getting assistant: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception verifying connection: {str(e)}")
        return False

def main():
    """Main function"""
    print("🚀 Fixing Vapi Assistant Tool Connection...")
    
    # Check current state
    print("\n1. Checking current configuration...")
    is_connected = verify_tool_connection()
    
    if not is_connected:
        print("\n2. Updating assistant configuration...")
        result = update_assistant_with_tool()
        
        if result:
            print("\n3. Verifying updated configuration...")
            verify_tool_connection()
    else:
        print("✅ Tool is already properly connected!")

if __name__ == "__main__":
    main()
