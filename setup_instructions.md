# Magentic-One General Agent System Setup

## Quick Start

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser dependencies (required for WebSurfer)
playwright install --with-deps chromium
```

### 2. Environment Setup
Your API keys are already configured in the `.env` file:
- ✅ OpenAI API Key (for GPT-4o)
- ✅ Deepgram API Key (for future voice features)

### 3. Run the System
```bash
python magentic_one_setup.py
```

## What You Get

Your general agent system includes these specialized agents working behind the scenes:

- **🧠 Orchestrator**: Plans tasks and coordinates other agents
- **🌐 WebSurfer**: Browses websites and extracts information
- **📁 FileSurfer**: Reads and navigates local files/directories
- **💻 Coder**: Writes and analyzes code
- **⚡ ComputerTerminal**: Executes code safely

## Security Features

- **Code Approval**: All code execution requires your approval
- **Docker Ready**: Can use Docker for isolated code execution
- **Safe Defaults**: Secure configuration out of the box

## Example Tasks

Try these to test your system:
1. "Research the latest AI developments and summarize them"
2. "Write a Python script to analyze data from a CSV file"
3. "Help me plan a trip to Japan with current information"
4. "Create a simple web scraper for product prices"

## Next Steps

- **Add Docker**: For enhanced security, install Docker and the system will automatically use it
- **Customize Agents**: Modify the agent configurations for your specific needs
- **Add Voice**: Use your Deepgram API key to add voice capabilities later
- **Scale Up**: Deploy to cloud infrastructure when ready

## Troubleshooting

- **Import Errors**: Run `pip install -r requirements.txt` again
- **Browser Issues**: Run `playwright install --with-deps chromium`
- **API Errors**: Check your `.env` file has the correct API keys
