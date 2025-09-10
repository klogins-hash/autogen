#!/usr/bin/env python3
"""
Vapi MCP Client for Magentic-One Integration
Connects Magentic-One to Vapi.ai via Model Context Protocol
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional
import docker
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MagenticOneVapiClient:
    """MCP client that bridges Vapi.ai with Magentic-One"""
    
    def __init__(self, vapi_token: str):
        self.vapi_token = vapi_token
        self.docker_client = docker.from_env()
        self.magentic_container_name = "magentic-one-agent"
        
    async def connect_to_vapi_mcp(self):
        """Connect to Vapi's remote MCP server"""
        # Use npx mcp-remote to connect to Vapi's hosted MCP server
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "mcp-remote", 
                "https://mcp.vapi.ai/mcp",
                "--header",
                f"Authorization: Bearer {self.vapi_token}"
            ],
            env={"VAPI_TOKEN": self.vapi_token}
        )
        
        self.session = await stdio_client(server_params)
        return self.session
    
    def query_magentic_one(self, query: str) -> str:
        """Send query to Magentic-One container and get response"""
        try:
            # Get the running Magentic-One container
            container = self.docker_client.containers.get(self.magentic_container_name)
            
            # Execute the query in the container
            # Using python -c to send the query to the running agent
            exec_command = f'''python -c "
import sys
sys.path.append('/app')
from magentic_one_helper import process_query
result = process_query('{query}')
print(result)
"'''
            
            result = container.exec_run(exec_command, workdir="/app")
            
            if result.exit_code == 0:
                return result.output.decode('utf-8').strip()
            else:
                return f"Error executing query: {result.output.decode('utf-8')}"
                
        except docker.errors.NotFound:
            return f"Magentic-One container '{self.magentic_container_name}' not found. Please ensure it's running."
        except Exception as e:
            return f"Error connecting to Magentic-One: {str(e)}"
    
    async def list_assistants(self):
        """List all Vapi assistants"""
        async with self.session as session:
            result = await session.call_tool("list_assistants", {})
            return self._parse_tool_response(result)
    
    async def create_call(self, assistant_id: str, phone_number: str, customer_phone: str):
        """Create an outbound call using Vapi"""
        async with self.session as session:
            result = await session.call_tool("create_call", {
                "assistantId": assistant_id,
                "customer": {
                    "number": customer_phone
                }
            })
            return self._parse_tool_response(result)
    
    async def get_call_status(self, call_id: str):
        """Get status of a specific call"""
        async with self.session as session:
            result = await session.call_tool("get_call", {"id": call_id})
            return self._parse_tool_response(result)
    
    def _parse_tool_response(self, response):
        """Parse MCP tool response"""
        if not response or not hasattr(response, 'content'):
            return response
            
        for item in response.content:
            if item.type == 'text':
                try:
                    return json.loads(item.text)
                except json.JSONDecodeError:
                    return item.text
        return response

class VapiMagenticBridge:
    """Main bridge class that handles Vapi-Magentic integration"""
    
    def __init__(self, vapi_token: str):
        self.client = MagenticOneVapiClient(vapi_token)
        self.session = None
    
    async def start(self):
        """Initialize the MCP connection"""
        print("🔌 Connecting to Vapi MCP server...")
        self.session = await self.client.connect_to_vapi_mcp()
        print("✅ Connected to Vapi MCP server")
        
        # List available tools
        tools = await self.session.list_tools()
        print(f"📋 Available Vapi tools: {len(tools.tools)}")
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description}")
    
    async def handle_voice_query(self, query: str) -> str:
        """Process a voice query through Magentic-One"""
        print(f"🎤 Processing voice query: {query}")
        
        # Send to Magentic-One for processing
        response = self.client.query_magentic_one(query)
        
        print(f"🤖 Magentic-One response: {response}")
        return response
    
    async def make_call(self, assistant_id: str, customer_phone: str):
        """Initiate a call using Vapi"""
        print(f"📞 Initiating call to {customer_phone} with assistant {assistant_id}")
        
        async with self.session as session:
            result = await session.call_tool("create_call", {
                "assistantId": assistant_id,
                "customer": {
                    "number": customer_phone
                }
            })
            
            call_data = self.client._parse_tool_response(result)
            print(f"✅ Call created: {call_data}")
            return call_data
    
    async def list_my_assistants(self):
        """List all available assistants"""
        async with self.session as session:
            result = await session.call_tool("list_assistants", {})
            assistants = self.client._parse_tool_response(result)
            
            print("🤖 Available Vapi Assistants:")
            if isinstance(assistants, list):
                for assistant in assistants:
                    print(f"  - {assistant.get('name', 'Unnamed')} (ID: {assistant.get('id')})")
            return assistants
    
    async def close(self):
        """Close the MCP connection"""
        if self.session:
            await self.session.close()
            print("🔌 Disconnected from Vapi MCP server")

async def main():
    """Main function to demonstrate the integration"""
    
    # Get Vapi token from environment
    vapi_token = os.getenv("VAPI_API_KEY")
    if not vapi_token:
        print("❌ Error: VAPI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Initialize the bridge
    bridge = VapiMagenticBridge(vapi_token)
    
    try:
        # Start the connection
        await bridge.start()
        
        # List assistants
        assistants = await bridge.list_my_assistants()
        
        # Example: Process a voice query
        test_query = "What's the weather like today in San Francisco?"
        response = await bridge.handle_voice_query(test_query)
        print(f"\n🎯 Final response: {response}")
        
        # Example: Make a call (uncomment to test)
        # if assistants and len(assistants) > 0:
        #     assistant_id = assistants[0]['id']
        #     await bridge.make_call(assistant_id, "+1234567890")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await bridge.close()

if __name__ == "__main__":
    asyncio.run(main())
