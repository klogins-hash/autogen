#!/usr/bin/env node
/**
 * Vapi MCP Service Starter - Runs as a persistent service
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import Docker from 'dockerode';
import dotenv from 'dotenv';

dotenv.config();

class VapiMCPService {
  constructor(vapiToken) {
    this.vapiToken = vapiToken;
    this.docker = new Docker();
    this.magenticContainer = 'magentic-one-agent';
    this.mcpClient = null;
    this.assistants = [];
  }

  async connect() {
    console.log('🔌 Connecting to Vapi MCP server...');
    
    this.mcpClient = new Client({
      name: 'vapi-magentic-service',
      version: '1.0.0',
    });

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

    await this.mcpClient.connect(transport);
    console.log('✅ Connected to Vapi MCP server');

    // List tools and assistants
    const tools = await this.mcpClient.listTools();
    console.log(`📋 Available tools: ${tools.tools.length}`);
    
    await this.loadAssistants();
    return true;
  }

  async loadAssistants() {
    const response = await this.mcpClient.callTool({
      name: 'list_assistants',
      arguments: {},
    });
    
    this.assistants = this.parseToolResponse(response);
    
    if (Array.isArray(this.assistants)) {
      console.log(`📋 Found ${this.assistants.length} assistants:`);
      this.assistants.forEach(assistant => {
        console.log(`  - ${assistant.name || 'Unnamed'} (${assistant.id})`);
      });
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

  async queryMagenticOne(query) {
    console.log(`🤖 Processing query: ${query}`);
    
    try {
      const container = this.docker.getContainer(this.magenticContainer);
      
      // Simple test query - replace with actual Magentic-One integration
      const exec = await container.exec({
        Cmd: ['python', '-c', `print("Magentic-One processed: ${query.replace(/"/g, '\\"')}")`],
        AttachStdout: true,
        AttachStderr: true,
      });

      const stream = await exec.start({ hijack: true, stdin: false });
      
      return new Promise((resolve, reject) => {
        let output = '';
        
        stream.on('data', (chunk) => {
          output += chunk.toString();
        });

        stream.on('end', () => {
          const result = output.trim() || `Processed: ${query}`;
          console.log(`💬 Response: ${result}`);
          resolve(result);
        });

        setTimeout(() => {
          resolve(`Timeout processing: ${query}`);
        }, 10000);
      });

    } catch (error) {
      const errorMsg = `Error: ${error.message}`;
      console.error('❌', errorMsg);
      return errorMsg;
    }
  }

  async testCall() {
    if (!this.assistants || this.assistants.length === 0) {
      console.log('❌ No assistants available for testing');
      return;
    }

    const assistantId = this.assistants[0].id;
    console.log(`📞 Testing call creation with assistant ${assistantId}`);
    
    try {
      const response = await this.mcpClient.callTool({
        name: 'create_call',
        arguments: {
          assistantId: assistantId,
          customer: {
            number: "+1234567890" // Test number
          }
        },
      });
      
      const callData = this.parseToolResponse(response);
      console.log('✅ Test call created:', callData);
      return callData;
    } catch (error) {
      console.error('❌ Call test failed:', error.message);
      return null;
    }
  }

  async runTests() {
    console.log('\n🧪 Running integration tests...');
    
    // Test 1: Query processing
    await this.queryMagenticOne("What's the weather like today?");
    
    // Test 2: List phone numbers
    try {
      const phoneResponse = await this.mcpClient.callTool({
        name: 'list_phone_numbers',
        arguments: {},
      });
      const phoneNumbers = this.parseToolResponse(phoneResponse);
      console.log(`📱 Phone numbers available: ${Array.isArray(phoneNumbers) ? phoneNumbers.length : 0}`);
    } catch (error) {
      console.log('📱 Phone numbers: Not configured yet');
    }
    
    console.log('\n✅ Integration tests completed');
  }

  async close() {
    if (this.mcpClient) {
      await this.mcpClient.close();
      console.log('🔌 Disconnected from Vapi MCP server');
    }
  }
}

async function main() {
  const vapiToken = process.env.VAPI_API_KEY || '867ac81c-f57e-49ae-9003-25c88de12a15';
  
  if (!vapiToken) {
    console.error('❌ Error: VAPI_API_KEY environment variable required');
    process.exit(1);
  }

  const service = new VapiMCPService(vapiToken);

  try {
    console.log('🚀 Starting Vapi MCP Service...');
    await service.connect();
    
    // Run tests
    await service.runTests();
    
    console.log('\n🎯 Service is running and ready for voice calls!');
    console.log('🔑 Assistant ID: e820f3e6-7a17-432e-be14-5bf5cbf6e611');
    console.log('📞 Configure phone number in Vapi dashboard to enable calling');
    
    // Keep service running
    console.log('\n⏳ Service running... Press Ctrl+C to stop');
    
    // Graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n🛑 Shutting down service...');
      await service.close();
      process.exit(0);
    });
    
    // Keep alive
    setInterval(() => {
      console.log(`⚡ Service alive - ${new Date().toISOString()}`);
    }, 60000); // Log every minute
    
  } catch (error) {
    console.error('❌ Service error:', error.message);
    await service.close();
    process.exit(1);
  }
}

main();
