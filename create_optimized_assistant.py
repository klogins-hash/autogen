#!/usr/bin/env python3
"""
Create Optimized Vapi Assistant for Magentic-One
Based on Vapi documentation best practices and optimization guidelines
"""

import requests
import json

# Vapi API configuration
VAPI_API_KEY = "867ac81c-f57e-49ae-9003-25c88de12a15"
VAPI_BASE_URL = "https://api.vapi.ai"
TOOL_ID = "bd815daa-5d73-4141-99e4-76374575f4e1"  # Created tool ID

headers = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json"
}

def create_optimized_assistant():
    """Create optimized Vapi assistant following best practices"""
    
    assistant_config = {
        "name": "Magentic-One General Agent",
        "firstMessage": "Hello! I'm your Magentic-One general agent. I can help you with research, coding, web browsing, file operations, and complex multi-step tasks. What would you like me to help you with today?",
        
        # Optimized model configuration
        "model": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,  # Balanced creativity and consistency
            "maxTokens": 250,    # Concise responses for voice
            "messages": [
                {
                    "role": "system",
                    "content": """[Identity]
You are a helpful and intelligent general agent powered by Magentic-One, a multi-agent system with specialized capabilities. You can handle complex tasks through coordination of multiple specialized agents including web browsing, coding, file operations, and research.

[Style]
- Be conversational and natural in speech
- Keep responses concise and focused for voice interaction
- Use simple, clear language avoiding technical jargon
- Show enthusiasm and helpfulness
- Ask clarifying questions when needed

[Response Guidelines]
- Limit responses to 2-3 sentences for voice clarity
- When handling complex tasks, use the query_magentic_one function
- Always confirm understanding before proceeding with complex requests
- Provide status updates for long-running tasks

[Task & Goals]
1. Listen to user requests and determine complexity
2. For simple questions, respond directly and conversationally
3. For complex tasks (research, coding, multi-step processes), use query_magentic_one function
4. Guide users through the process and provide clear updates
5. Handle errors gracefully with helpful suggestions

[Tool Usage]
- Use query_magentic_one for: research tasks, coding requests, web browsing needs, file operations, multi-step workflows, complex analysis
- Pass the complete user request as the 'query' parameter
- Wait for the tool response before continuing the conversation

[Error Handling]
If you encounter unclear requests, ask specific clarifying questions. If technical issues arise, explain simply and offer alternatives. Always maintain a helpful, patient tone."""
                }
            ]
        },
        
        # Optimized voice configuration
        "voice": {
            "provider": "openai",
            "voiceId": "alloy",
            "speed": 1.1        # Slightly faster for efficiency
        },
        
        # Advanced speech configuration for optimal conversation flow
        "startSpeakingPlan": {
            "waitSeconds": 0.6,  # Slightly longer wait for complex responses
            "smartEndpointingEnabled": True,
            "smartEndpointingPlan": {
                "provider": "livekit",  # Best for English conversations
                "waitFunction": "300 + 6000 * x"  # Optimized timing curve
            },
            "transcriptionEndpointingPlan": {
                "onPunctuationSeconds": 0.8,
                "onNoPunctuationSeconds": 1.2,
                "onNumberSeconds": 0.5
            }
        },
        
        "stopSpeakingPlan": {
            "numWords": 2,       # Allow brief acknowledgments
            "voiceSeconds": 0.3, # Quick response to interruptions
            "backoffSeconds": 0.8 # Resume speaking promptly
        },
        
        # Function tools configuration
        "functions": [
            {
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
            }
        ],
        
        # Server URL for function calls
        "server": {
            "url": "https://206.189.203.174:8081/webhook"
        },
        
        # Advanced configuration
        "backgroundSound": "off",
        "backchannelingEnabled": True,  # Natural conversation flow
        "backgroundDenoisingEnabled": True,
        "modelOutputInMessagesEnabled": True,
        
        # Response optimization
        "responseDelaySeconds": 0.2,  # Slight delay for natural feel
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
            "end-of-call-report",
            "function-call",
            "hang",
            "speech-update",
            "status-update",
            "transcript"
        ]
    }
    
    response = requests.post(
        f"{VAPI_BASE_URL}/assistant",
        headers=headers,
        json=assistant_config
    )
    
    if response.status_code == 201:
        assistant = response.json()
        print("✅ Optimized assistant created successfully!")
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
        "fallbackDestination": {
            "type": "number",
            "phoneNumber": "+1234567890"  # Replace with actual fallback
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
        return phone
    else:
        print(f"❌ Error creating phone number: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def main():
    print("🚀 Creating Optimized Vapi Assistant for Magentic-One...")
    print(f"Using API Key: {VAPI_API_KEY[:8]}...")
    print(f"Tool ID: {TOOL_ID}")
    print()
    
    # Create optimized assistant
    print("🎯 Creating optimized assistant with best practices...")
    assistant = create_optimized_assistant()
    
    if assistant:
        print()
        print("📞 Creating phone number...")
        phone = create_phone_number(assistant['id'])
        
        if phone:
            print()
            print("🎉 Optimized Setup Complete!")
            print("=" * 60)
            print(f"Assistant ID: {assistant['id']}")
            print(f"Assistant Name: {assistant['name']}")
            print(f"Phone Number: {phone['number']}")
            print(f"Tool ID: {TOOL_ID}")
            print("=" * 60)
            print()
            print("🎯 Optimization Features Enabled:")
            print("✅ Smart endpointing with LiveKit")
            print("✅ Optimized speech timing")
            print("✅ Advanced transcription with Deepgram Nova-2")
            print("✅ Natural conversation flow")
            print("✅ Background denoising")
            print("✅ Structured prompting with sections")
            print("✅ Tool integration for complex tasks")
            print()
            print("Next steps:")
            print("1. Deploy the webhook server to handle function calls")
            print("2. Test the assistant by calling the phone number")
            print("3. Monitor performance and adjust settings as needed")
        else:
            print("⚠️  Assistant created but phone number setup failed")
            print(f"You can manually configure a phone number for assistant: {assistant['id']}")
    else:
        print("❌ Failed to create optimized assistant")

if __name__ == "__main__":
    main()
