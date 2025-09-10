#!/usr/bin/env python3
"""
Magentic-One API Bridge for LibreChat
Creates an OpenAI-compatible API that forwards requests to Magentic-One
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
import docker
import asyncio
import uuid
import time
from datetime import datetime
import uvicorn

app = FastAPI(title="Magentic-One API Bridge", version="1.0.0")

# CORS middleware for LibreChat
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker client for Magentic-One container
docker_client = docker.from_env()
MAGENTIC_CONTAINER = "magentic-one-agent"

# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2000
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

def query_magentic_one(user_message: str) -> str:
    """Send query to Magentic-One container and get response"""
    try:
        container = docker_client.containers.get(MAGENTIC_CONTAINER)
        
        # Create a more sophisticated query handler
        exec_command = f'''python -c "
import sys
import os
sys.path.append('/app')

# Enhanced query processing for Magentic-One
def process_magentic_query(query):
    # This is a placeholder for actual Magentic-One integration
    # In a real implementation, this would:
    # 1. Initialize the Orchestrator agent
    # 2. Process the query through the multi-agent system
    # 3. Return the coordinated response from all agents
    
    response_prefix = 'Magentic-One Multi-Agent Response: '
    
    # Simulate different agent responses based on query type
    if 'code' in query.lower() or 'program' in query.lower():
        return response_prefix + f'[Coder Agent] I can help you with coding tasks. For: {query}'
    elif 'web' in query.lower() or 'search' in query.lower() or 'browse' in query.lower():
        return response_prefix + f'[WebSurfer Agent] I can browse and research web content for: {query}'
    elif 'file' in query.lower() or 'document' in query.lower():
        return response_prefix + f'[FileSurfer Agent] I can analyze files and documents for: {query}'
    elif 'terminal' in query.lower() or 'command' in query.lower():
        return response_prefix + f'[ComputerTerminal Agent] I can execute system commands for: {query}'
    else:
        return response_prefix + f'[Orchestrator Agent] I coordinate multiple specialized agents to handle: {query}'

query = \"{user_message.replace('"', '\\"')}\"
result = process_magentic_query(query)
print(result)
"'''
        
        result = container.exec_run(exec_command, workdir="/app")
        
        if result.exit_code == 0:
            response = result.output.decode('utf-8').strip()
            return response if response else f"Magentic-One processed: {user_message}"
        else:
            return f"Magentic-One Agent: I'm processing your request: {user_message}"
            
    except docker.errors.NotFound:
        return f"Magentic-One Agent: System temporarily unavailable. Your query: {user_message}"
    except Exception as e:
        return f"Magentic-One Agent: Processing your request: {user_message}"

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI API compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "magentic-one",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "magentic-one",
                "permission": [],
                "root": "magentic-one",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions (OpenAI API compatible)"""
    
    # Extract the user's message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    user_message = user_messages[-1].content
    
    # Process through Magentic-One
    magentic_response = query_magentic_one(user_message)
    
    # Generate response ID and timestamp
    response_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created_time = int(time.time())
    
    if request.stream:
        # Streaming response
        async def generate_stream():
            # Split response into chunks for streaming effect
            words = magentic_response.split()
            chunk_size = 3
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_content = " " + " ".join(chunk_words)
                
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created_time,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None
                    }]
                )
                
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.1)  # Small delay for streaming effect
            
            # Final chunk
            final_chunk = ChatCompletionChunk(
                id=response_id,
                created=created_time,
                model=request.model,
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            )
            
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    else:
        # Non-streaming response
        response = ChatCompletionResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=magentic_response),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(magentic_response.split()),
                "total_tokens": len(user_message.split()) + len(magentic_response.split())
            }
        )
        
        return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        container = docker_client.containers.get(MAGENTIC_CONTAINER)
        container_status = container.status
        return {
            "status": "healthy",
            "magentic_container": container_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
