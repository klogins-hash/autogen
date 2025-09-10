#!/usr/bin/env python3
"""
Magentic-One Helper for MCP Integration
This script runs inside the Magentic-One container to process queries
"""

import sys
import os
import json
from typing import Dict, Any

def process_query(query: str) -> str:
    """
    Process a query using Magentic-One agents
    This function should be called from within the Magentic-One container
    """
    try:
        # Import Magentic-One components
        from autogen_magentic_one.agents.orchestrator import Orchestrator
        from autogen_magentic_one.agents.web_surfer import WebSurfer
        from autogen_magentic_one.agents.file_surfer import FileSurfer
        from autogen_magentic_one.agents.coder import Coder
        from autogen_magentic_one.agents.computer_terminal import ComputerTerminal
        from autogen_core.components.models import OpenAIChatCompletionClient
        
        # Initialize model client
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize agents
        orchestrator = Orchestrator(model_client=model_client)
        web_surfer = WebSurfer(model_client=model_client)
        file_surfer = FileSurfer(model_client=model_client)
        coder = Coder(model_client=model_client)
        computer_terminal = ComputerTerminal(model_client=model_client)
        
        # Process the query through the orchestrator
        result = orchestrator.process_task(query)
        
        return str(result)
        
    except ImportError as e:
        return f"Error importing Magentic-One components: {str(e)}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

def main():
    """Main function for standalone execution"""
    if len(sys.argv) < 2:
        print("Usage: python magentic_one_helper.py '<query>'")
        sys.exit(1)
    
    query = sys.argv[1]
    result = process_query(query)
    print(result)

if __name__ == "__main__":
    main()
