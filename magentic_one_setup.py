#!/usr/bin/env python3
"""
Magentic-One General Agent System
A unified interface that presents multiple specialized agents as a single general agent.
"""

import asyncio
import os
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def approval_func(request: ApprovalRequest) -> ApprovalResponse:
    """Simple approval function that requests user input before code execution."""
    print(f"\n🔧 Code Execution Request:")
    print(f"{'='*50}")
    print(f"{request.code}")
    print(f"{'='*50}")
    
    while True:
        user_input = input("\n❓ Approve code execution? (y/n/s for show more): ").strip().lower()
        if user_input == 'y':
            return ApprovalResponse(approved=True, reason="User approved the code execution")
        elif user_input == 'n':
            return ApprovalResponse(approved=False, reason="User denied the code execution")
        elif user_input == 's':
            print(f"\n📋 Full request details:")
            print(f"Code: {request.code}")
            print(f"Reason: {getattr(request, 'reason', 'No reason provided')}")
        else:
            print("Please enter 'y' for yes, 'n' for no, or 's' to show more details")


async def create_general_agent():
    """Create and configure the Magentic-One general agent system."""
    
    # Initialize OpenAI client with GPT-4o (recommended for Magentic-One)
    client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create Magentic-One system with security controls
    agent_system = MagenticOne(
        client=client,
        hil_mode=False,  # Set to True if you want human-in-the-loop for all interactions
        approval_func=approval_func  # Require approval for code execution
    )
    
    return agent_system, client


async def run_general_agent():
    """Main function to run the general agent system."""
    
    print("🚀 Initializing Magentic-One General Agent System...")
    print("📋 This system includes:")
    print("   • Orchestrator (planning and coordination)")
    print("   • WebSurfer (web browsing and interaction)")
    print("   • FileSurfer (file system operations)")
    print("   • Coder (code generation and analysis)")
    print("   • ComputerTerminal (code execution)")
    print("   • Security: Code execution requires approval")
    print()
    
    # Create the agent system
    agent_system, client = await create_general_agent()
    
    try:
        # Example tasks to demonstrate capabilities
        tasks = [
            "Research the latest developments in AI agents and create a brief summary",
            "Write a Python script to analyze a CSV file and generate basic statistics",
            "Help me understand the current weather in San Francisco and suggest activities",
        ]
        
        print("🎯 Example tasks you can try:")
        for i, task in enumerate(tasks, 1):
            print(f"   {i}. {task}")
        print()
        
        # Interactive mode
        while True:
            task = input("💬 Enter your task (or 'quit' to exit): ").strip()
            
            if task.lower() in ['quit', 'exit', 'q']:
                break
                
            if not task:
                continue
                
            print(f"\n🤖 Processing: {task}")
            print("="*60)
            
            try:
                # Run the task through the agent system
                result = await Console(agent_system.run_stream(task=task))
                print("\n✅ Task completed!")
                
            except KeyboardInterrupt:
                print("\n⏸️  Task interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error occurred: {str(e)}")
                continue
                
    finally:
        # Clean up
        await client.close()
        print("\n👋 General Agent System shut down successfully")


if __name__ == "__main__":
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in the .env file")
        exit(1)
    
    # Run the general agent system
    asyncio.run(run_general_agent())
