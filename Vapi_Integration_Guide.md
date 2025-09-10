# Vapi.ai Integration Guide: Using FastAPI as Voice Layer for Magentic-One

This guide provides step-by-step instructions to configure Vapi.ai as a professional voice interface for your existing Magentic-One general agent system.

## Overview

Instead of using the custom FastAPI voice frontend, you'll use Vapi.ai's professional voice infrastructure while keeping your Magentic-One backend. This provides:

- **Professional Voice Quality**: Enterprise-grade speech-to-text and text-to-speech
- **Phone Number Support**: Inbound/outbound calling capabilities
- **Managed Infrastructure**: No need to manage voice processing servers
- **Advanced Features**: Call recording, analytics, and monitoring

## Architecture

```
Caller → Vapi.ai Voice Processing → Your FastAPI Webhook → Magentic-One → Response
```

## Prerequisites

1. **Vapi.ai Account**: Sign up at [dashboard.vapi.ai](https://dashboard.vapi.ai)
2. **Running Magentic-One**: Your Docker container at `206.189.203.174`
3. **Public Webhook URL**: Your FastAPI server accessible from the internet

---

## Step 1: Create Vapi.ai Account

### 1.1 Sign Up
1. Go to [https://dashboard.vapi.ai](https://dashboard.vapi.ai)
2. Click **"Sign Up"**
3. Complete registration with your email
4. Verify your email address

### 1.2 Get API Key
1. Navigate to **"API Keys"** in the dashboard
2. Click **"Create New Key"**
3. Copy and save your API key (starts with `vapi_`)

---

## Step 2: Create FastAPI Webhook Endpoint

### 2.1 Create Vapi Webhook Server

Create a new file `/root/vapi-webhook/vapi_webhook.py`:

```python
#!/usr/bin/env python3
"""
Vapi.ai Webhook Server for Magentic-One Integration
Receives Vapi events and forwards to Magentic-One system
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
import docker
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vapi Webhook for Magentic-One")
docker_client = docker.from_env()

async def query_magentic_one(text: str) -> str:
    """Send query to Magentic-One container and get response"""
    try:
        # Get the running Magentic-One container
        container = docker_client.containers.get("magentic-one-agent")
        
        # Create a script to interact with the agent
        script = f'''
import asyncio
import sys
import os
sys.path.append('/app')

async def query_agent():
    try:
        # Import after path is set
        from magentic_one_setup import create_general_agent
        
        agent_system, client = await create_general_agent()
        
        # Query the agent system
        result = await agent_system.run(task="{text.replace('"', '\\"')}")
        
        # Extract response
        if hasattr(result, 'messages') and result.messages:
            response = result.messages[-1].content
        else:
            response = str(result)
            
        await client.close()
        return response
    except Exception as e:
        return f"Error processing request: {{str(e)}}"

if __name__ == "__main__":
    result = asyncio.run(query_agent())
    print(result)
'''
        
        # Execute the script in the container
        exec_result = container.exec_run(
            f"python -c '{script}'",
            workdir="/app"
        )
        
        response = exec_result.output.decode('utf-8').strip()
        return response if response else "I'm sorry, I couldn't process that request."
        
    except Exception as e:
        print(f"Magentic-One query error: {e}")
        return "I'm experiencing technical difficulties. Please try again."

@app.post("/webhook")
async def vapi_webhook(request: Request):
    """Handle Vapi webhook events"""
    try:
        body = await request.json()
        message = body.get("message", {})
        message_type = message.get("type")
        
        print(f"Received Vapi event: {message_type}")
        
        # Handle different event types
        if message_type == "assistant-request":
            # Return assistant configuration for incoming calls
            return JSONResponse({
                "assistant": {
                    "firstMessage": "Hello! I'm your Magentic-One general agent. How can I help you today?",
                    "model": {
                        "provider": "openai",
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful general agent powered by Magentic-One. You can help with research, coding, web browsing, file operations, and complex multi-step tasks. Keep responses conversational and ask clarifying questions when needed."
                            }
                        ]
                    },
                    "voice": {
                        "provider": "openai",
                        "voiceId": "alloy"
                    }
                }
            })
            
        elif message_type == "tool-calls":
            # Handle function calls from the assistant
            tool_calls = message.get("toolCallList", [])
            results = []
            
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                parameters = tool_call.get("parameters", {})
                
                if tool_name == "query_magentic_one":
                    # Extract the user's query
                    user_query = parameters.get("query", "")
                    
                    # Query Magentic-One system
                    response = await query_magentic_one(user_query)
                    
                    results.append({
                        "name": tool_name,
                        "toolCallId": tool_id,
                        "result": json.dumps({"response": response})
                    })
            
            return JSONResponse({"results": results})
            
        elif message_type == "status-update":
            # Log status updates
            status = message.get("status")
            print(f"Call status: {status}")
            
        elif message_type == "end-of-call-report":
            # Log call completion
            ended_reason = message.get("endedReason")
            print(f"Call ended: {ended_reason}")
            
        # Return 200 OK for all events
        return JSONResponse({"status": "received"})
        
    except Exception as e:
        print(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Vapi Webhook for Magentic-One"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 2.2 Create Requirements File

Create `/root/vapi-webhook/requirements.txt`:

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
docker==7.1.0
python-dotenv==1.0.0
python-multipart==0.0.6
```

### 2.3 Deploy Webhook Server

```bash
# On your server (206.189.203.174)
mkdir -p /root/vapi-webhook
cd /root/vapi-webhook

# Upload the files (vapi_webhook.py and requirements.txt)
# Then install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the webhook server
python vapi_webhook.py
```

---

## Step 3: Configure Vapi Assistant

### 3.1 Create Assistant via Dashboard

1. **Login to Vapi Dashboard**: [https://dashboard.vapi.ai](https://dashboard.vapi.ai)

2. **Navigate to Assistants**: Click **"Assistants"** in the left sidebar

3. **Create New Assistant**: Click **"Create Assistant"**

4. **Basic Configuration**:
   - **Name**: `Magentic-One General Agent`
   - **First Message**: `Hello! I'm your Magentic-One general agent. How can I help you today?`

5. **Model Configuration**:
   - **Provider**: `OpenAI`
   - **Model**: `gpt-4o`
   - **System Message**: 
     ```
     You are a helpful general agent powered by Magentic-One. You can help with research, coding, web browsing, file operations, and complex multi-step tasks. When users ask for complex tasks, use the query_magentic_one function to process their request through the specialized agent system. Keep responses conversational and ask clarifying questions when needed.
     ```

6. **Voice Configuration**:
   - **Provider**: `OpenAI`
   - **Voice**: `alloy` (or your preference: nova, echo, fable, onyx, shimmer)

### 3.2 Add Function Tool

1. **Go to Functions Tab**: In your assistant configuration

2. **Add Function**: Click **"Add Function"**

3. **Function Configuration**:
   - **Name**: `query_magentic_one`
   - **Description**: `Send complex queries to the Magentic-One general agent system for processing`
   - **Parameters**:
     ```json
     {
       "type": "object",
       "properties": {
         "query": {
           "type": "string",
           "description": "The user's query or task to be processed by Magentic-One"
         }
       },
       "required": ["query"]
     }
     ```

4. **Server URL**: `http://206.189.203.174:8081/webhook`

### 3.3 Configure Server URL

1. **Advanced Tab**: In your assistant configuration

2. **Server URL**: `http://206.189.203.174:8081/webhook`

3. **Save Assistant**: Click **"Save"** to create your assistant

---

## Step 4: Set Up Phone Number (Optional)

### 4.1 Get Phone Number

1. **Navigate to Phone Numbers**: In Vapi dashboard

2. **Buy Number**: Click **"Buy Phone Number"**

3. **Select Country/Area**: Choose your preferred location

4. **Purchase**: Complete the purchase process

### 4.2 Configure Phone Number

1. **Select Your Number**: Click on your purchased number

2. **Assistant Configuration**:
   - **Assistant**: Select `Magentic-One General Agent`
   - **Server URL**: `http://206.189.203.174:8081/webhook`

3. **Save Configuration**

---

## Step 5: Test Your Integration

### 5.1 Test via Dashboard

1. **Go to Assistants**: Select your `Magentic-One General Agent`

2. **Test Button**: Click **"Test"** in the top right

3. **Start Conversation**: Click the microphone and speak

4. **Test Complex Query**: Try: *"Research the latest developments in AI agents and create a summary"*

### 5.2 Test via Phone (if configured)

1. **Call Your Number**: Dial your Vapi phone number

2. **Wait for Greeting**: You should hear the first message

3. **Test Functionality**: Ask for complex tasks that require multiple agents

### 5.3 Monitor Logs

```bash
# On your server, check webhook logs
tail -f /var/log/vapi-webhook.log

# Check Magentic-One container logs
docker logs magentic-one-agent -f
```

---

## Step 6: Advanced Configuration

### 6.1 Add Authentication (Recommended)

1. **Create Custom Credential**: In Vapi dashboard
   - Go to **"Custom Credentials"**
   - Create **"Bearer Token"** credential
   - Add your API token
   - Note the credential ID

2. **Update Assistant**: Add credential ID to server configuration

### 6.2 Configure Call Recording

1. **Assistant Settings**: Go to advanced settings

2. **Recording**: Enable call recording

3. **Storage**: Configure where recordings are stored

### 6.3 Set Up Analytics

1. **Dashboard Analytics**: View call metrics and performance

2. **Webhook Events**: Log important events for monitoring

---

## Troubleshooting

### Common Issues

1. **Webhook Not Receiving Events**:
   - Verify your server URL is publicly accessible
   - Check firewall settings on port 8081
   - Test webhook endpoint with curl

2. **Magentic-One Not Responding**:
   - Verify Docker container is running: `docker ps`
   - Check container logs: `docker logs magentic-one-agent`
   - Ensure proper API keys are configured

3. **Voice Quality Issues**:
   - Try different voice providers (OpenAI, ElevenLabs)
   - Adjust voice settings in assistant configuration
   - Check network connectivity

### Testing Commands

```bash
# Test webhook endpoint
curl -X POST http://206.189.203.174:8081/webhook \
  -H "Content-Type: application/json" \
  -d '{"message": {"type": "status-update", "status": "test"}}'

# Test Magentic-One container
docker exec magentic-one-agent python -c "print('Container is responsive')"

# Check webhook server status
curl http://206.189.203.174:8081/health
```

---

## Next Steps

1. **Customize Voice**: Experiment with different voice providers and settings
2. **Add More Functions**: Create additional tools for specific use cases
3. **Monitor Performance**: Set up logging and analytics
4. **Scale**: Consider load balancing for high-volume usage
5. **Security**: Implement proper authentication and rate limiting

---

## Support Resources

- **Vapi Documentation**: [https://docs.vapi.ai](https://docs.vapi.ai)
- **Vapi Discord**: [https://discord.gg/vapi](https://discord.gg/vapi)
- **API Reference**: [https://api.vapi.ai/api](https://api.vapi.ai/api)
- **Magentic-One Docs**: Your local AutoGen documentation

Your Magentic-One system now has professional voice capabilities through Vapi.ai!
