#!/usr/bin/env node
/**
 * Vapi MCP Integration for Magentic-One
 * Uses Vapi's MCP server with Node.js MCP SDK
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { spawn } from 'child_process';
import Docker from 'dockerode';
import dotenv from 'dotenv';

dotenv.config();

class VapiMagenticBridge {
  constructor(vapiToken) {
    this.vapiToken = vapiToken;
    this.docker = new Docker();
    this.magenticContainer = 'magentic-one-agent';
    this.mcpClient = null;
  }

  async connect() {
    console.log('🔌 Connecting to Vapi MCP server...');
    
    // Initialize MCP client
    this.mcpClient = new Client({
      name: 'vapi-magentic-bridge',
      version: '1.0.0',
    });

    // Create transport
    const transport = new StreamableHTTPClientTransport(
      new URL('https://mcp.vapi.ai/mcp'),
      {
        requestInit: {
          headers: {
            Authorization: `Bearer ${this.vapiToken}`,
          },
        },
      }
    );

    // Connect
    await this.mcpClient.connect(transport);
    console.log('✅ Connected to Vapi MCP server');

    // List available tools
    const tools = await this.mcpClient.listTools();
    console.log(`📋 Available tools: ${tools.tools.length}`);
    tools.tools.forEach(tool => {
      console.log(`  - ${tool.name}: ${tool.description}`);
    });

    return true;
  }

  async queryMagenticOne(query) {
    console.log(`🤖 Sending to Magentic-One: ${query}`);
    
    try {
      const container = this.docker.getContainer(this.magenticContainer);
      
      // Execute query in container
      const exec = await container.exec({
        Cmd: ['python', '-c', `
import sys
import os
sys.path.append('/app')

# Simple query processing - replace with actual Magentic-One integration
def process_query(query):
    # This is a placeholder - integrate with actual Magentic-One agents
    return f"Magentic-One processed: {query}"

query = "${query.replace(/"/g, '\\"')}"
result = process_query(query)
print(result)
        `],
        AttachStdout: true,
        AttachStderr: true,
      });

      const stream = await exec.start({ hijack: true, stdin: false });
      
      return new Promise((resolve, reject) => {
        let output = '';
        let error = '';
        
        stream.on('data', (chunk) => {
          const data = chunk.toString();
          if (data.includes('stdout')) {
            output += data.replace(/.*stdout.*/, '');
          } else if (data.includes('stderr')) {
            error += data.replace(/.*stderr.*/, '');
          }
        });

        stream.on('end', () => {
          if (error) {
            reject(new Error(error));
          } else {
            resolve(output.trim() || `Processed query: ${query}`);
          }
        });
      });

    } catch (error) {
      console.error('❌ Error querying Magentic-One:', error.message);
      return `Error: ${error.message}`;
    }
  }

  parseToolResponse(response) {
    if (!response?.content) return response;
    
    const textItem = response.content.find(item => item.type === 'text');
    if (textItem?.text) {
      try {
        return JSON.parse(textItem.text);
      } catch {
        return textItem.text;
      }
    }
    return response;
  }

  async listAssistants() {
    console.log('📋 Listing Vapi assistants...');
    
    const response = await this.mcpClient.callTool({
      name: 'list_assistants',
      arguments: {},
    });
    
    const assistants = this.parseToolResponse(response);
    
    if (Array.isArray(assistants)) {
      console.log(`Found ${assistants.length} assistants:`);
      assistants.forEach(assistant => {
        console.log(`  - ${assistant.name || 'Unnamed'} (${assistant.id})`);
      });
    }
    
    return assistants;
  }

  async createCall(assistantId, customerPhone) {
    console.log(`📞 Creating call to ${customerPhone} with assistant ${assistantId}`);
    
    const response = await this.mcpClient.callTool({
      name: 'create_call',
      arguments: {
        assistantId: assistantId,
        customer: {
          number: customerPhone
        }
      },
    });
    
    const callData = this.parseToolResponse(response);
    console.log('✅ Call created:', callData);
    return callData;
  }

  async getCallStatus(callId) {
    const response = await this.mcpClient.callTool({
      name: 'get_call',
      arguments: { id: callId },
    });
    
    return this.parseToolResponse(response);
  }

  async close() {
    if (this.mcpClient) {
      await this.mcpClient.close();
      console.log('🔌 Disconnected from Vapi MCP server');
    }
  }
}

class VapiMagenticService {
  constructor(vapiToken) {
    this.bridge = new VapiMagenticBridge(vapiToken);
    this.assistants = [];
  }

  async start() {
    console.log('🚀 Starting Vapi-Magentic Integration Service...');
    
    await this.bridge.connect();
    this.assistants = await this.bridge.listAssistants();
    
    console.log('✅ Service started successfully');
    return this.assistants;
  }

  async processVoiceQuery(query) {
    console.log(`🎤 Processing voice query: ${query}`);
    
    // Send to Magentic-One for processing
    const response = await this.bridge.queryMagenticOne(query);
    
    console.log(`💬 Response: ${response}`);
    return response;
  }

  async makeCall(customerPhone) {
    if (!this.assistants || this.assistants.length === 0) {
      throw new Error('No assistants available');
    }

    const assistantId = this.assistants[0].id;
    return await this.bridge.createCall(assistantId, customerPhone);
  }

  async stop() {
    await this.bridge.close();
    console.log('🛑 Service stopped');
  }
}

// Interactive CLI
async function runInteractive(service) {
  const { createInterface } = await import('readline');
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('\n🎯 Interactive Mode:');
  console.log('Commands:');
  console.log('  call <phone> - Make a call');
  console.log('  quit - Exit');
  console.log('  Any other text - Process as query');

  const askQuestion = () => {
    rl.question('\n> ', async (input) => {
      const command = input.trim();

      if (command.toLowerCase() === 'quit') {
        rl.close();
        return;
      }

      if (command.startsWith('call ')) {
        const phone = command.substring(5).trim();
        try {
          await service.makeCall(phone);
        } catch (error) {
          console.error('❌ Call failed:', error.message);
        }
      } else if (command) {
        try {
          const response = await service.processVoiceQuery(command);
          console.log(`\n💬 Final Response: ${response}`);
        } catch (error) {
          console.error('❌ Query failed:', error.message);
        }
      }

      askQuestion();
    });
  };

  askQuestion();
}

async function main() {
  const vapiToken = process.env.VAPI_API_KEY || '867ac81c-f57e-49ae-9003-25c88de12a15';
  
  if (!vapiToken) {
    console.error('❌ Error: VAPI_API_KEY environment variable required');
    process.exit(1);
  }

  const service = new VapiMagenticService(vapiToken);

  try {
    await service.start();
    
    // Run interactive mode
    await runInteractive(service);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  } finally {
    await service.stop();
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down...');
  process.exit(0);
});

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
