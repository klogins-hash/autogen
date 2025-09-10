#!/usr/bin/env python3
"""
Voice Frontend for Magentic-One General Agent System
FastAPI + WebSocket server with Deepgram STT and OpenAI TTS
"""

import asyncio
import json
import os
import base64
import io
from typing import Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import websockets
from deepgram import DeepgramClient, PrerecordedOptions, LiveTranscriptionEvents, LiveOptions
from openai import OpenAI
import docker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Magentic-One Voice Frontend")

# Initialize clients
deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
docker_client = docker.from_env()

class VoiceManager:
    def __init__(self):
        self.active_connections = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(message))

voice_manager = VoiceManager()

async def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe audio using Deepgram"""
    try:
        # Configure Deepgram options
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=False,
        )
        
        # Create audio source
        audio_source = {"buffer": audio_data, "mimetype": "audio/wav"}
        
        # Transcribe
        response = deepgram.listen.prerecorded.v("1").transcribe_file(audio_source, options)
        
        # Extract transcript
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript.strip()
        
    except Exception as e:
        print(f"Deepgram transcription error: {e}")
        return ""

async def generate_speech(text: str) -> bytes:
    """Generate speech using OpenAI TTS"""
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="wav"
        )
        return response.content
    except Exception as e:
        print(f"OpenAI TTS error: {e}")
        return b""

async def query_magentic_one(text: str) -> str:
    """Send query to Magentic-One container and get response"""
    try:
        # Get the running Magentic-One container
        container = docker_client.containers.get("magentic-one-agent")
        
        # Create a temporary Python script to interact with the agent
        script = f'''
import asyncio
import sys
import os
sys.path.append('/app')

from magentic_one_setup import create_general_agent

async def query_agent():
    try:
        agent_system, client = await create_general_agent()
        
        # Simple query without interactive UI
        from autogen_agentchat.base import TaskResult
        result = await agent_system.run(task="{text}")
        
        # Extract the final response
        if hasattr(result, 'messages') and result.messages:
            response = result.messages[-1].content
        else:
            response = str(result)
            
        await client.close()
        return response
    except Exception as e:
        return f"Error: {{str(e)}}"

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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await voice_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                # Decode base64 audio data
                audio_data = base64.b64decode(message["data"])
                
                # Send status update
                await voice_manager.send_message(client_id, {
                    "type": "status",
                    "message": "Transcribing..."
                })
                
                # Transcribe audio
                transcript = await transcribe_audio(audio_data)
                
                if transcript:
                    # Send transcript to client
                    await voice_manager.send_message(client_id, {
                        "type": "transcript",
                        "text": transcript
                    })
                    
                    # Send status update
                    await voice_manager.send_message(client_id, {
                        "type": "status", 
                        "message": "Processing with Magentic-One..."
                    })
                    
                    # Query Magentic-One
                    response_text = await query_magentic_one(transcript)
                    
                    # Send text response
                    await voice_manager.send_message(client_id, {
                        "type": "response",
                        "text": response_text
                    })
                    
                    # Generate speech
                    await voice_manager.send_message(client_id, {
                        "type": "status",
                        "message": "Generating speech..."
                    })
                    
                    audio_response = await generate_speech(response_text)
                    
                    if audio_response:
                        # Send audio response
                        audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                        await voice_manager.send_message(client_id, {
                            "type": "audio_response",
                            "data": audio_b64
                        })
                    
                    # Clear status
                    await voice_manager.send_message(client_id, {
                        "type": "status",
                        "message": ""
                    })
                
    except WebSocketDisconnect:
        voice_manager.disconnect(client_id)

@app.get("/", response_class=HTMLResponse)
async def get_voice_interface():
    """Serve the voice chat interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magentic-One Voice Assistant</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .status {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            min-height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
        }
        .record-button {
            background: #ff4757;
            color: white;
            border: none;
            border-radius: 50%;
            width: 100px;
            height: 100px;
            font-size: 24px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 20px;
        }
        .record-button:hover {
            transform: scale(1.1);
            box-shadow: 0 10px 20px rgba(255, 71, 87, 0.3);
        }
        .record-button.recording {
            background: #2ed573;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        .transcript, .response {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            text-align: left;
            border-left: 4px solid #667eea;
        }
        .response {
            border-left-color: #2ed573;
        }
        .controls {
            margin: 20px 0;
        }
        .volume-control {
            margin: 10px 0;
        }
        .volume-control input {
            width: 200px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Magentic-One Voice Assistant</h1>
        <p>Click the microphone to start talking to your general agent</p>
        
        <div class="status" id="status">Ready to listen...</div>
        
        <div class="controls">
            <button class="record-button" id="recordButton">🎤</button>
        </div>
        
        <div class="volume-control">
            <label for="volume">Response Volume:</label>
            <input type="range" id="volume" min="0" max="1" step="0.1" value="0.7">
        </div>
        
        <div id="transcript" class="transcript" style="display: none;">
            <strong>You said:</strong> <span id="transcriptText"></span>
        </div>
        
        <div id="response" class="response" style="display: none;">
            <strong>Magentic-One:</strong> <span id="responseText"></span>
        </div>
    </div>

    <script>
        class VoiceAssistant {
            constructor() {
                this.ws = null;
                this.mediaRecorder = null;
                this.audioChunks = [];
                this.isRecording = false;
                this.clientId = Math.random().toString(36).substring(7);
                
                this.recordButton = document.getElementById('recordButton');
                this.status = document.getElementById('status');
                this.transcript = document.getElementById('transcript');
                this.transcriptText = document.getElementById('transcriptText');
                this.response = document.getElementById('response');
                this.responseText = document.getElementById('responseText');
                this.volumeControl = document.getElementById('volume');
                
                this.init();
            }
            
            async init() {
                await this.connectWebSocket();
                await this.setupAudio();
                this.setupEventListeners();
            }
            
            async connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                };
                
                this.ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    setTimeout(() => this.connectWebSocket(), 3000);
                };
            }
            
            async setupAudio() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    this.mediaRecorder = new MediaRecorder(stream);
                    
                    this.mediaRecorder.ondataavailable = (event) => {
                        this.audioChunks.push(event.data);
                    };
                    
                    this.mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                        this.audioChunks = [];
                        this.sendAudio(audioBlob);
                    };
                } catch (error) {
                    console.error('Error accessing microphone:', error);
                    this.status.textContent = 'Error: Could not access microphone';
                }
            }
            
            setupEventListeners() {
                this.recordButton.addEventListener('click', () => {
                    if (this.isRecording) {
                        this.stopRecording();
                    } else {
                        this.startRecording();
                    }
                });
            }
            
            startRecording() {
                if (this.mediaRecorder && this.mediaRecorder.state === 'inactive') {
                    this.isRecording = true;
                    this.recordButton.classList.add('recording');
                    this.recordButton.textContent = '🛑';
                    this.status.textContent = 'Listening... Click to stop';
                    this.mediaRecorder.start();
                }
            }
            
            stopRecording() {
                if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                    this.isRecording = false;
                    this.recordButton.classList.remove('recording');
                    this.recordButton.textContent = '🎤';
                    this.status.textContent = 'Processing...';
                    this.mediaRecorder.stop();
                }
            }
            
            async sendAudio(audioBlob) {
                const arrayBuffer = await audioBlob.arrayBuffer();
                const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: 'audio',
                        data: base64Audio
                    }));
                }
            }
            
            handleMessage(message) {
                switch (message.type) {
                    case 'status':
                        this.status.textContent = message.message || 'Ready to listen...';
                        break;
                        
                    case 'transcript':
                        this.transcriptText.textContent = message.text;
                        this.transcript.style.display = 'block';
                        break;
                        
                    case 'response':
                        this.responseText.textContent = message.text;
                        this.response.style.display = 'block';
                        break;
                        
                    case 'audio_response':
                        this.playAudio(message.data);
                        break;
                }
            }
            
            playAudio(base64Audio) {
                const audioData = atob(base64Audio);
                const audioArray = new Uint8Array(audioData.length);
                for (let i = 0; i < audioData.length; i++) {
                    audioArray[i] = audioData.charCodeAt(i);
                }
                
                const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.volume = parseFloat(this.volumeControl.value);
                audio.play();
            }
        }
        
        // Initialize the voice assistant when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new VoiceAssistant();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Magentic-One Voice Frontend"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
