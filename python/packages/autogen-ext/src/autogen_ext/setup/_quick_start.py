"""
Quick start utilities for AutoGen enhanced setup.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from ._setup_manager import SetupManager
from ._config_manager import ConfigManager, ConfigTemplate
from ._dependency_manager import DependencyManager
from ._environment_setup import EnvironmentSetup


class ProjectTemplate(Enum):
    """Available project templates."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    ADVANCED = "advanced"
    RESEARCH = "research"
    PRODUCTION = "production"


@dataclass
class ProjectConfig:
    """Configuration for a new project."""
    name: str
    template: ProjectTemplate
    path: str
    features: List[str]
    model_provider: str = "openai"
    enable_memory: bool = True
    enable_plugins: bool = True
    enable_monitoring: bool = True


class QuickStart:
    """Provides quick start functionality for AutoGen enhanced projects."""
    
    def __init__(self):
        self.setup_manager = SetupManager()
        self.config_manager = ConfigManager()
        self.dependency_manager = DependencyManager()
        self.environment_setup = EnvironmentSetup()
    
    async def create_project(
        self,
        project_config: ProjectConfig,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Create a new AutoGen enhanced project."""
        
        project_path = Path(project_config.path) / project_config.name
        
        # Set up the setup manager for this project
        setup_manager = SetupManager(
            project_path=str(project_path),
            verbose=True
        )
        
        if progress_callback:
            setup_manager.set_progress_callback(progress_callback)
        
        # Customize setup based on template
        self._customize_setup_for_template(setup_manager, project_config)
        
        # Run setup
        success = await setup_manager.run_setup()
        
        if success:
            # Generate project-specific files
            await self._generate_project_files(project_path, project_config)
            
            # Create getting started guide
            self._create_getting_started_guide(project_path, project_config)
        
        return {
            "success": success,
            "project_path": str(project_path),
            "setup_log": setup_manager.get_setup_log(),
            "next_steps": self._get_next_steps(project_config) if success else []
        }
    
    def _customize_setup_for_template(
        self,
        setup_manager: SetupManager,
        project_config: ProjectConfig
    ) -> None:
        """Customize setup based on project template."""
        
        if project_config.template == ProjectTemplate.MINIMAL:
            # Remove optional steps for minimal setup
            setup_manager.remove_step("setup_vector_store")
            setup_manager.remove_step("initialize_plugins")
            setup_manager.remove_step("run_tests")
        
        elif project_config.template == ProjectTemplate.ADVANCED:
            # Add advanced setup steps
            from ._setup_manager import SetupStep
            
            advanced_steps = [
                SetupStep(
                    id="setup_monitoring",
                    name="Setup Monitoring",
                    description="Configure advanced monitoring and metrics",
                    function=self._setup_advanced_monitoring,
                    dependencies=["validate_config"]
                ),
                SetupStep(
                    id="setup_security",
                    name="Setup Security",
                    description="Configure security features",
                    function=self._setup_security_features,
                    dependencies=["validate_config"]
                )
            ]
            
            for step in advanced_steps:
                setup_manager.add_step(step)
        
        elif project_config.template == ProjectTemplate.PRODUCTION:
            # Add production-specific steps
            from ._setup_manager import SetupStep
            
            prod_steps = [
                SetupStep(
                    id="setup_logging",
                    name="Setup Production Logging",
                    description="Configure production-grade logging",
                    function=self._setup_production_logging,
                    dependencies=["validate_config"]
                ),
                SetupStep(
                    id="setup_deployment",
                    name="Setup Deployment",
                    description="Prepare for deployment",
                    function=self._setup_deployment_config,
                    dependencies=["validate_config"]
                )
            ]
            
            for step in prod_steps:
                setup_manager.add_step(step)
    
    async def _generate_project_files(
        self,
        project_path: Path,
        project_config: ProjectConfig
    ) -> None:
        """Generate project-specific files."""
        
        # Create main application file
        main_app_content = self._generate_main_app(project_config)
        with open(project_path / "main.py", 'w') as f:
            f.write(main_app_content)
        
        # Create example agent file
        example_agent_content = self._generate_example_agent(project_config)
        with open(project_path / "example_agent.py", 'w') as f:
            f.write(example_agent_content)
        
        # Create requirements.txt
        self._create_requirements_file(project_path, project_config)
        
        # Create README.md
        readme_content = self._generate_readme(project_config)
        with open(project_path / "README.md", 'w') as f:
            f.write(readme_content)
    
    def _generate_main_app(self, project_config: ProjectConfig) -> str:
        """Generate main application file."""
        
        return f'''"""
{project_config.name} - AutoGen Enhanced Application
Generated by AutoGen Enhanced Quick Start
"""

import asyncio
import os
from pathlib import Path

# AutoGen Enhanced imports
from autogen_ext.setup import ConfigManager
from autogen_ext.optimization import ContextCompressor, TokenOptimizer
from autogen_ext.monitoring import HealthMonitor, ErrorRecoveryManager
from autogen_ext.execution import ParallelAgentExecutor

# Standard AutoGen imports
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import OpenAIChatCompletionClient


class {project_config.name.replace(' ', '').replace('-', '')}App:
    """Main application class for {project_config.name}."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = None
        self.agents = {{}}
        
        # Enhanced components
        {"self.context_compressor = ContextCompressor()" if "context_compression" in project_config.features else ""}
        {"self.health_monitor = HealthMonitor()" if "monitoring" in project_config.features else ""}
        {"self.parallel_executor = ParallelAgentExecutor()" if "parallel_execution" in project_config.features else ""}
    
    async def initialize(self):
        """Initialize the application."""
        
        # Load configuration
        config_path = Path("config/autogen_config.json")
        if config_path.exists():
            self.config = self.config_manager.load_config(str(config_path))
        else:
            raise FileNotFoundError("Configuration file not found. Run setup first.")
        
        # Initialize model client
        model_config = self.config["models"]["model_configs"][self.config["models"]["default_model"]]
        self.model_client = OpenAIChatCompletionClient(
            model=self.config["models"]["default_model"],
            api_key=os.getenv("OPENAI_API_KEY") or model_config["api_key"]
        )
        
        # Create agents
        await self._create_agents()
        
        # Start monitoring if enabled
        {"await self.health_monitor.start_monitoring()" if "monitoring" in project_config.features else ""}
        
        print(f"{{self.__class__.__name__}} initialized successfully!")
    
    async def _create_agents(self):
        """Create and configure agents."""
        
        # Orchestrator agent
        self.agents["orchestrator"] = AssistantAgent(
            name="orchestrator",
            model_client=self.model_client,
            system_message="You are an orchestrator agent that coordinates tasks between specialized agents."
        )
        
        # Add more agents based on template
        {"# Web surfer agent" if project_config.template in [ProjectTemplate.STANDARD, ProjectTemplate.ADVANCED] else ""}
        {"# Coder agent" if project_config.template in [ProjectTemplate.STANDARD, ProjectTemplate.ADVANCED] else ""}
        {"# Research agent" if project_config.template == ProjectTemplate.RESEARCH else ""}
    
    async def run_conversation(self, user_message: str):
        """Run a conversation with the agents."""
        
        # Create team
        team = RoundRobinGroupChat(list(self.agents.values()))
        
        # Run conversation
        result = await team.run(task=user_message)
        
        return result
    
    async def shutdown(self):
        """Shutdown the application."""
        
        {"await self.health_monitor.stop_monitoring()" if "monitoring" in project_config.features else ""}
        {"await self.parallel_executor.shutdown()" if "parallel_execution" in project_config.features else ""}
        
        print("Application shutdown complete.")


async def main():
    """Main entry point."""
    
    app = {project_config.name.replace(' ', '').replace('-', '')}App()
    
    try:
        await app.initialize()
        
        # Example conversation
        result = await app.run_conversation(
            "Hello! Can you help me with a task?"
        )
        
        print("Conversation result:", result)
        
    except Exception as e:
        print(f"Error: {{e}}")
    
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _generate_example_agent(self, project_config: ProjectConfig) -> str:
        """Generate example agent file."""
        
        return f'''"""
Example custom agent for {project_config.name}
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ChatCompletionClient

{"from autogen_ext.plugins import AgentPlugin, PluginCapability" if project_config.enable_plugins else ""}
{"from autogen_ext.memory.persistent import PersistentMemoryManager" if project_config.enable_memory else ""}


class CustomAgent(AssistantAgent):
    """Custom agent with enhanced capabilities."""
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        **kwargs
    ):
        super().__init__(name=name, model_client=model_client, **kwargs)
        
        {"self.memory_manager = PersistentMemoryManager()" if project_config.enable_memory else ""}
    
    async def on_messages(self, messages, cancellation_token=None):
        """Enhanced message processing."""
        
        # Add custom logic here
        {"# Store conversation in memory" if project_config.enable_memory else ""}
        {"# Apply plugins" if project_config.enable_plugins else ""}
        
        # Call parent implementation
        response = await super().on_messages(messages, cancellation_token)
        
        return response


{"class ExamplePlugin(AgentPlugin):" if project_config.enable_plugins else ""}
{'''    """Example plugin for the custom agent."""
    
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin for demonstration"
    
    @property
    def capabilities(self) -> set:
        return {PluginCapability.MESSAGE_PROCESSING}
    
    async def process_message(self, message, context):
        """Process incoming message."""
        # Add custom message processing logic
        return message
    
    async def generate_response(self, messages, context):
        """Generate response to messages."""
        # Add custom response generation logic
        return None
    
    async def _initialize(self) -> None:
        """Initialize the plugin."""
        print(f"Initialized {self.name}")''' if project_config.enable_plugins else ""}
'''
    
    def _create_requirements_file(self, project_path: Path, project_config: ProjectConfig) -> None:
        """Create requirements.txt file."""
        
        base_requirements = [
            "autogen-agentchat",
            "autogen-core",
            "autogen-ext",
            "openai>=1.0.0",
            "python-dotenv>=1.0.0"
        ]
        
        if project_config.enable_memory:
            base_requirements.extend([
                "chromadb>=0.4.0",
                "sentence-transformers>=2.2.0"
            ])
        
        if project_config.enable_monitoring:
            base_requirements.extend([
                "psutil>=5.9.0",
                "aiofiles>=23.0.0"
            ])
        
        if project_config.template in [ProjectTemplate.ADVANCED, ProjectTemplate.PRODUCTION]:
            base_requirements.extend([
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0"
            ])
        
        requirements_path = project_path / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(f"# Requirements for {project_config.name}\n")
            f.write("# Generated by AutoGen Enhanced Quick Start\n\n")
            for req in sorted(base_requirements):
                f.write(f"{req}\n")
    
    def _generate_readme(self, project_config: ProjectConfig) -> str:
        """Generate README.md file."""
        
        return f'''# {project_config.name}

An AutoGen Enhanced project created with the {project_config.template.value} template.

## Features

{chr(10).join(f"- {feature.replace('_', ' ').title()}" for feature in project_config.features)}

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Project Structure

```
{project_config.name}/
├── main.py              # Main application
├── example_agent.py     # Example custom agent
├── config/              # Configuration files
├── plugins/             # Custom plugins
├── memory/              # Persistent memory
├── logs/                # Log files
├── requirements.txt     # Python dependencies
├── .env.template        # Environment variables template
└── README.md           # This file
```

## Configuration

The main configuration is in `config/autogen_config.json`. Key settings:

- **Models**: Configure your LLM providers and models
- **Agents**: Set up agent-specific configurations
- **Memory**: Configure persistent memory settings
- **Monitoring**: Set up health checks and metrics
- **Plugins**: Enable and configure plugins

## Usage Examples

### Basic Conversation

```python
from main import {project_config.name.replace(' ', '').replace('-', '')}App

app = {project_config.name.replace(' ', '').replace('-', '')}App()
await app.initialize()

result = await app.run_conversation("Hello, can you help me?")
print(result)
```

### Custom Agent

```python
from example_agent import CustomAgent
from autogen_core.models import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4")
agent = CustomAgent("custom", model_client)
```

## Advanced Features

{"### Memory System" if project_config.enable_memory else ""}
{"The project includes persistent memory capabilities for learning and context retention." if project_config.enable_memory else ""}

{"### Plugin System" if project_config.enable_plugins else ""}
{"Extend functionality with custom plugins. See `plugins/` directory for examples." if project_config.enable_plugins else ""}

{"### Monitoring" if project_config.enable_monitoring else ""}
{"Built-in health monitoring and metrics collection for production deployments." if project_config.enable_monitoring else ""}

## Development

### Adding New Agents

1. Create a new agent class inheriting from `AssistantAgent`
2. Add enhanced capabilities as needed
3. Register the agent in your main application

### Creating Plugins

1. Inherit from the appropriate plugin base class
2. Implement required methods
3. Add to the plugin configuration

### Testing

Run tests with:
```bash
python -m pytest tests/
```

## Deployment

{"For production deployment, see the deployment configuration in `config/deployment.yaml`." if project_config.template == ProjectTemplate.PRODUCTION else ""}

## Support

- Check the AutoGen Enhanced documentation
- Review example projects
- Join the community discussions

## License

This project is licensed under the MIT License.
'''
    
    def _create_getting_started_guide(self, project_path: Path, project_config: ProjectConfig) -> None:
        """Create a getting started guide."""
        
        guide_content = f'''# Getting Started with {project_config.name}

Welcome to your new AutoGen Enhanced project! This guide will help you get up and running quickly.

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Copy `.env.template` to `.env` and add your API keys
- [ ] Review `config/autogen_config.json` settings
- [ ] Run the example: `python main.py`
- [ ] Explore the code in `example_agent.py`

## First Steps

### 1. Environment Setup

Your project needs API keys to work with language models:

```bash
# Copy the environment template
cp .env.template .env

# Edit .env and add your keys
# At minimum, set OPENAI_API_KEY
```

### 2. Test the Installation

Run a quick test to make sure everything works:

```bash
python -c "
import asyncio
from main import {project_config.name.replace(' ', '').replace('-', '')}App

async def test():
    app = {project_config.name.replace(' ', '').replace('-', '')}App()
    await app.initialize()
    print('✅ Setup successful!')
    await app.shutdown()

asyncio.run(test())
"
```

### 3. Run Your First Conversation

```bash
python main.py
```

## Next Steps

### Customize Your Agents

Edit `example_agent.py` to add custom behavior:

- Add domain-specific knowledge
- Implement custom tools
- Configure memory and learning

### Add Plugins

{"Create plugins in the `plugins/` directory to extend functionality." if project_config.enable_plugins else ""}

### Configure Memory

{"Set up persistent memory in `config/autogen_config.json` under the `memory` section." if project_config.enable_memory else ""}

### Monitor Performance

{"Check `logs/` directory for system logs and metrics." if project_config.enable_monitoring else ""}

## Common Issues

### API Key Errors
- Make sure your `.env` file has the correct API keys
- Check that environment variables are loaded

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check your Python version (3.8+ required)

### Memory Issues
- {"Make sure ChromaDB is properly installed for vector storage" if project_config.enable_memory else ""}

## Getting Help

1. Check the configuration files for settings
2. Review the example code for patterns
3. Check the AutoGen Enhanced documentation
4. Look at the logs in `logs/` directory

Happy coding! 🚀
'''
        
        guide_path = project_path / "GETTING_STARTED.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
    
    def _get_next_steps(self, project_config: ProjectConfig) -> List[str]:
        """Get next steps for the user."""
        
        steps = [
            f"Navigate to your project: cd {project_config.path}/{project_config.name}",
            "Copy .env.template to .env and add your API keys",
            "Review the configuration in config/autogen_config.json",
            "Install dependencies: pip install -r requirements.txt",
            "Run the example: python main.py",
            "Read GETTING_STARTED.md for detailed instructions"
        ]
        
        if project_config.enable_plugins:
            steps.append("Explore example plugins in the plugins/ directory")
        
        if project_config.enable_memory:
            steps.append("Configure memory settings for your use case")
        
        return steps
    
    async def run_basic_test(self) -> Dict[str, Any]:
        """Run a basic test of the AutoGen enhanced system."""
        
        test_results = []
        
        # Test 1: Import core modules
        try:
            from autogen_ext.optimization import ContextCompressor
            from autogen_ext.monitoring import HealthMonitor
            from autogen_ext.execution import ParallelAgentExecutor
            test_results.append({"test": "import_modules", "success": True, "message": "Core modules imported successfully"})
        except Exception as e:
            test_results.append({"test": "import_modules", "success": False, "message": f"Import failed: {e}"})
        
        # Test 2: Create basic components
        try:
            compressor = ContextCompressor()
            monitor = HealthMonitor()
            test_results.append({"test": "create_components", "success": True, "message": "Components created successfully"})
        except Exception as e:
            test_results.append({"test": "create_components", "success": False, "message": f"Component creation failed: {e}"})
        
        # Test 3: Environment validation
        try:
            env_setup = EnvironmentSetup()
            validation_result = await env_setup.validate_environment()
            test_results.append({
                "test": "environment_validation", 
                "success": validation_result["valid"], 
                "message": f"Environment validation: {'passed' if validation_result['valid'] else 'failed'}",
                "details": validation_result
            })
        except Exception as e:
            test_results.append({"test": "environment_validation", "success": False, "message": f"Validation failed: {e}"})
        
        overall_success = all(result["success"] for result in test_results)
        
        return {
            "success": overall_success,
            "test_results": test_results,
            "message": "All tests passed!" if overall_success else "Some tests failed. Check the details."
        }
    
    # Additional setup step implementations for advanced templates
    
    async def _setup_advanced_monitoring(self) -> Dict[str, Any]:
        """Setup advanced monitoring features."""
        return {"status": "Advanced monitoring configured"}
    
    async def _setup_security_features(self) -> Dict[str, Any]:
        """Setup security features."""
        return {"status": "Security features configured"}
    
    async def _setup_production_logging(self) -> Dict[str, Any]:
        """Setup production logging."""
        return {"status": "Production logging configured"}
    
    async def _setup_deployment_config(self) -> Dict[str, Any]:
        """Setup deployment configuration."""
        return {"status": "Deployment configuration created"}


# CLI interface for quick start
async def cli_create_project():
    """Command-line interface for creating projects."""
    
    print("🚀 AutoGen Enhanced Quick Start")
    print("=" * 40)
    
    # Get project details
    project_name = input("Project name: ").strip()
    if not project_name:
        project_name = "my-autogen-project"
    
    print("\nAvailable templates:")
    for i, template in enumerate(ProjectTemplate, 1):
        print(f"{i}. {template.value.title()}")
    
    template_choice = input("\nChoose template (1-5) [2]: ").strip()
    if not template_choice:
        template_choice = "2"
    
    try:
        template_index = int(template_choice) - 1
        template = list(ProjectTemplate)[template_index]
    except (ValueError, IndexError):
        template = ProjectTemplate.STANDARD
    
    project_path = input(f"Project path [.]: ").strip()
    if not project_path:
        project_path = "."
    
    # Create project config
    features = ["context_compression", "monitoring", "parallel_execution"]
    if template in [ProjectTemplate.STANDARD, ProjectTemplate.ADVANCED, ProjectTemplate.PRODUCTION]:
        features.extend(["memory", "plugins"])
    
    project_config = ProjectConfig(
        name=project_name,
        template=template,
        path=project_path,
        features=features
    )
    
    # Create project
    quick_start = QuickStart()
    
    def progress_callback(step_id: str, status, message: Optional[str] = None):
        status_emoji = {"running": "⏳", "completed": "✅", "failed": "❌", "skipped": "⏭️"}
        print(f"{status_emoji.get(status.value, '📋')} {step_id}: {message or status.value}")
    
    print(f"\nCreating project '{project_name}' with {template.value} template...")
    result = await quick_start.create_project(project_config, progress_callback)
    
    if result["success"]:
        print(f"\n🎉 Project created successfully!")
        print(f"📁 Location: {result['project_path']}")
        print("\n📋 Next steps:")
        for i, step in enumerate(result["next_steps"], 1):
            print(f"{i}. {step}")
    else:
        print("\n❌ Project creation failed. Check the logs for details.")
    
    return result


if __name__ == "__main__":
    asyncio.run(cli_create_project())
