#!/usr/bin/env python3
"""
Fix Vapi Assistant Tool Connection
Updates the assistant to properly use the Magentic-One tool
"""

import requests
import json
import os

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
    
    # Updated assistant configuration with proper tool reference
    assistant_config = {
        "name": "Magentic-One General Agent",
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "maxTokens": 1000,
            "systemMessage": """You are the Magentic-One General Agent, a sophisticated AI assistant powered by a multi-agent system with specialized capabilities.

## Identity & Capabilities
You have access to multiple specialized agents working behind the scenes:
- **Web Research**: Browse and analyze web content
- **File Analysis**: Read and process documents  
- **Code Development**: Write, debug, and execute code
- **System Operations**: Terminal and system commands
- **Task Orchestration**: Complex multi-step workflows

## Communication Style
- **Concise & Clear**: Keep responses brief and actionable for voice interaction
- **Professional**: Maintain expertise while being approachable
- **Efficient**: Get to the point quickly, avoid unnecessary elaboration
- **Confident**: Provide definitive answers when possible

## Response Guidelines
- **Voice-Optimized**: Responses designed for spoken delivery
- **Action-Oriented**: Focus on what you can accomplish
- **No Filler**: Avoid "um," "well," or unnecessary qualifiers
- **Direct**: Answer questions directly before providing context

## Tool Usage
For complex queries requiring research, analysis, coding, or multi-step tasks, use the query_magentic_one function to leverage the full multi-agent system.

## Task Goals
1. **Understand** user requests quickly and accurately
2. **Process** complex tasks through specialized agents when needed  
3. **Deliver** clear, actionable responses optimized for voice
4. **Handle** errors gracefully with helpful alternatives

Remember: You represent a powerful general agent system. Be confident in your capabilities while staying concise for voice interaction."""
        },
        
        # Voice configuration
        "voice": {
            "provider": "openai",
            "voiceId": "alloy",
            "speed": 1.1,
            "fillerInjectionEnabled": False
        },
        
        # Speech timing optimization
        "startSpeakingPlan": {
            "waitSeconds": 0.6,
            "smartEndpointingEnabled": True,
            "smartEndpointingPlan": {
                "provider": "livekit",
                "waitFunction": "300 + 6000 * x"
            },
            "transcriptionEndpointingPlan": {
                "onPunctuationSeconds": 0.8,
                "onNoPunctuationSeconds": 1.2,
                "onNumberSeconds": 0.5
            }
        },
        
        "stopSpeakingPlan": {
            "numWords": 2,
            "voiceSeconds": 0.3,
            "backoffSeconds": 0.8
        },
        
        # Tool connection - Use the actual tool ID instead of inline function
        "toolIds": [TOOL_ID],
        
        # Advanced configuration
        "backgroundSound": "off",
        "backchannelingEnabled": True,
        "backgroundDenoisingEnabled": True,
        "modelOutputInMessagesEnabled": True,
        
        # Response optimization
        "responseDelaySeconds": 0.2,
        "llmRequestDelaySeconds": 0.1,
        
        # Transcription settings
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en",
            "smartFormat": True,
            "keywords": ["magneticone:1", "research:1", "coding:1", "analysis:1"]
        },
        
        # Analytics and monitoring
        "clientMessages": [
            "conversation-update",
            "function-call",
            "hang",
            "speech-update"
        ],
        
        "serverMessages": [
            "conversation-update", 
            "function-call",
            "hang",
            "speech-update"
        ]
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
    
    # First verify current state
    print("\n1. Checking current configuration...")
    is_connected = verify_tool_connection()
    
    if not is_connected:
        print("\n2. Updating assistant configuration...")
        result = update_assistant_with_tool()
        
        if result:
            print("\n3. Verifying updated configuration...")
            verify_tool_connection()
        else:
            print("❌ Failed to update assistant")
    else:
        print("✅ Tool is already properly connected!")

if __name__ == "__main__":
    main()
