#!/usr/bin/env python3
"""
Vapi MCP Integration for Magentic-One
Uses Vapi's MCP server to create a seamless voice-to-agent bridge
"""

import asyncio
import json
import os
import sys
import subprocess
import docker
from typing import Dict, Any, Optional, List

class VapiMCPBridge:
    """Bridge between Vapi MCP server and Magentic-One"""
    
    def __init__(self, vapi_token: str):
        self.vapi_token = vapi_token
        self.docker_client = docker.from_env()
        self.magentic_container_name = "magentic-one-agent"
        self.mcp_process = None
        
    async def start_mcp_connection(self):
        """Start connection to Vapi MCP server using npx mcp-remote"""
        print("🔌 Starting Vapi MCP connection...")
        
        # Set environment variables
        env = os.environ.copy()
        env["VAPI_TOKEN"] = self.vapi_token
        
        # Start the MCP remote connection
        cmd = [
            "npx", "mcp-remote",
            "https://mcp.vapi.ai/mcp",
            "--header", f"Authorization: Bearer {self.vapi_token}"
        ]
        
        try:
            self.mcp_process = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✅ MCP connection established")
            return True
        except Exception as e:
            print(f"❌ Failed to start MCP connection: {e}")
            return False
    
    def send_mcp_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server"""
        if not self.mcp_process:
            raise Exception("MCP connection not established")
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.mcp_process.stdin.write(request_json)
        self.mcp_process.stdin.flush()
        
        # Read response
        response_line = self.mcp_process.stdout.readline()
        if response_line:
            return json.loads(response_line.strip())
        
        return {"error": "No response from MCP server"}
    
    def query_magentic_one(self, query: str) -> str:
        """Send query to Magentic-One container"""
        try:
            container = self.docker_client.containers.get(self.magentic_container_name)
            
            # Copy helper script to container if not exists
            helper_script = "/Users/franksimpson/CascadeProjects/autogen/magentic_one_helper.py"
            if os.path.exists(helper_script):
                # Copy to container
                with open(helper_script, 'rb') as f:
                    container.put_archive('/app/', f.read())
            
            # Execute query
            exec_command = f'python /app/magentic_one_helper.py "{query}"'
            result = container.exec_run(exec_command, workdir="/app")
            
            if result.exit_code == 0:
                return result.output.decode('utf-8').strip()
            else:
                return f"Error: {result.output.decode('utf-8')}"
                
        except docker.errors.NotFound:
            return f"Magentic-One container '{self.magentic_container_name}' not found"
        except Exception as e:
            return f"Error connecting to Magentic-One: {str(e)}"
    
    def list_assistants(self) -> List[Dict[str, Any]]:
        """List all Vapi assistants via MCP"""
        try:
            response = self.send_mcp_request("tools/call", {
                "name": "list_assistants",
                "arguments": {}
            })
            
            if "result" in response:
                content = response["result"].get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        return json.loads(item["text"])
            
            return []
        except Exception as e:
            print(f"Error listing assistants: {e}")
            return []
    
    def create_call(self, assistant_id: str, customer_phone: str) -> Dict[str, Any]:
        """Create outbound call via MCP"""
        try:
            response = self.send_mcp_request("tools/call", {
                "name": "create_call",
                "arguments": {
                    "assistantId": assistant_id,
                    "customer": {
                        "number": customer_phone
                    }
                }
            })
            
            if "result" in response:
                content = response["result"].get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        return json.loads(item["text"])
            
            return response
        except Exception as e:
            print(f"Error creating call: {e}")
            return {"error": str(e)}
    
    def get_call_status(self, call_id: str) -> Dict[str, Any]:
        """Get call status via MCP"""
        try:
            response = self.send_mcp_request("tools/call", {
                "name": "get_call",
                "arguments": {"id": call_id}
            })
            
            if "result" in response:
                content = response["result"].get("content", [])
                for item in content:
                    if item.get("type") == "text":
                        return json.loads(item["text"])
            
            return response
        except Exception as e:
            print(f"Error getting call status: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close MCP connection"""
        if self.mcp_process:
            self.mcp_process.terminate()
            self.mcp_process.wait()
            print("🔌 MCP connection closed")

class VapiMagenticService:
    """Main service class for Vapi-Magentic integration"""
    
    def __init__(self, vapi_token: str):
        self.bridge = VapiMCPBridge(vapi_token)
        self.running = False
    
    async def start(self):
        """Start the integration service"""
        print("🚀 Starting Vapi-Magentic Integration Service...")
        
        # Start MCP connection
        if not await self.bridge.start_mcp_connection():
            raise Exception("Failed to establish MCP connection")
        
        # List available assistants
        assistants = self.bridge.list_assistants()
        print(f"📋 Found {len(assistants)} Vapi assistants:")
        for assistant in assistants:
            print(f"  - {assistant.get('name', 'Unnamed')} (ID: {assistant.get('id')})")
        
        self.running = True
        print("✅ Integration service started successfully")
        
        return assistants
    
    def process_voice_query(self, query: str) -> str:
        """Process voice query through Magentic-One"""
        print(f"🎤 Processing: {query}")
        
        # Send to Magentic-One
        response = self.bridge.query_magentic_one(query)
        
        print(f"🤖 Response: {response}")
        return response
    
    def initiate_call(self, assistant_id: str, customer_phone: str):
        """Initiate a call using Vapi"""
        print(f"📞 Calling {customer_phone} with assistant {assistant_id}")
        
        call_result = self.bridge.create_call(assistant_id, customer_phone)
        
        if "error" not in call_result:
            print(f"✅ Call initiated: {call_result.get('id', 'Unknown ID')}")
        else:
            print(f"❌ Call failed: {call_result['error']}")
        
        return call_result
    
    def stop(self):
        """Stop the integration service"""
        self.running = False
        self.bridge.close()
        print("🛑 Integration service stopped")

async def main():
    """Main function"""
    
    # Get Vapi token
    vapi_token = os.getenv("VAPI_API_KEY", "867ac81c-f57e-49ae-9003-25c88de12a15")
    if not vapi_token:
        print("❌ Error: VAPI_API_KEY environment variable required")
        sys.exit(1)
    
    # Initialize service
    service = VapiMagenticService(vapi_token)
    
    try:
        # Start the service
        assistants = await service.start()
        
        # Interactive mode
        print("\n🎯 Interactive Mode - Enter queries or commands:")
        print("Commands:")
        print("  'call <phone>' - Make a call using first assistant")
        print("  'quit' - Exit")
        print("  Any other text - Process as Magentic-One query")
        
        while service.running:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.startswith('call '):
                    phone = user_input[5:].strip()
                    if assistants:
                        assistant_id = assistants[0]['id']
                        service.initiate_call(assistant_id, phone)
                    else:
                        print("❌ No assistants available")
                elif user_input:
                    response = service.process_voice_query(user_input)
                    print(f"\n💬 Final Response: {response}")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        service.stop()

if __name__ == "__main__":
    asyncio.run(main())
