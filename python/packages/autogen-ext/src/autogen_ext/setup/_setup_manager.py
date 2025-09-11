"""
Setup manager for orchestrating the installation and configuration process.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import subprocess
import sys

from ._dependency_manager import DependencyManager, DependencyInfo
from ._config_manager import ConfigManager, ConfigTemplate
from ._environment_setup import EnvironmentSetup


class SetupStatus(Enum):
    """Status of setup steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SetupStep:
    """Represents a single setup step."""
    id: str
    name: str
    description: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    timeout: float = 300.0  # 5 minutes default
    status: SetupStatus = SetupStatus.PENDING
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None


class SetupManager:
    """Manages the complete setup process for AutoGen enhancements."""
    
    def __init__(
        self,
        project_path: str = ".",
        config_file: str = "autogen_config.json",
        verbose: bool = True
    ):
        self.project_path = Path(project_path)
        self.config_file = self.project_path / config_file
        self.verbose = verbose
        
        # Components
        self.dependency_manager = DependencyManager()
        self.config_manager = ConfigManager()
        self.environment_setup = EnvironmentSetup()
        
        # Setup steps
        self.steps: Dict[str, SetupStep] = {}
        self.execution_order: List[str] = []
        
        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.setup_log: List[Dict[str, Any]] = []
        
        # Initialize default steps
        self._initialize_default_steps()
    
    def _initialize_default_steps(self) -> None:
        """Initialize default setup steps."""
        
        default_steps = [
            SetupStep(
                id="validate_environment",
                name="Validate Environment",
                description="Check Python version and basic requirements",
                function=self._validate_environment
            ),
            SetupStep(
                id="create_directories",
                name="Create Directories",
                description="Create necessary project directories",
                function=self._create_directories
            ),
            SetupStep(
                id="install_dependencies",
                name="Install Dependencies",
                description="Install required Python packages",
                function=self._install_dependencies,
                dependencies=["validate_environment"]
            ),
            SetupStep(
                id="setup_vector_store",
                name="Setup Vector Store",
                description="Initialize ChromaDB for persistent memory",
                function=self._setup_vector_store,
                dependencies=["install_dependencies"],
                optional=True
            ),
            SetupStep(
                id="create_config",
                name="Create Configuration",
                description="Generate configuration files",
                function=self._create_config,
                dependencies=["create_directories"]
            ),
            SetupStep(
                id="validate_config",
                name="Validate Configuration",
                description="Validate generated configuration",
                function=self._validate_config,
                dependencies=["create_config"]
            ),
            SetupStep(
                id="initialize_plugins",
                name="Initialize Plugins",
                description="Set up plugin system",
                function=self._initialize_plugins,
                dependencies=["validate_config"],
                optional=True
            ),
            SetupStep(
                id="run_tests",
                name="Run Tests",
                description="Run basic functionality tests",
                function=self._run_tests,
                dependencies=["validate_config"],
                optional=True
            )
        ]
        
        for step in default_steps:
            self.add_step(step)
    
    def add_step(self, step: SetupStep) -> None:
        """Add a setup step."""
        self.steps[step.id] = step
        self._update_execution_order()
    
    def remove_step(self, step_id: str) -> bool:
        """Remove a setup step."""
        if step_id in self.steps:
            del self.steps[step_id]
            self._update_execution_order()
            return True
        return False
    
    def set_progress_callback(self, callback: Callable[[str, SetupStatus, Optional[str]], None]) -> None:
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    async def run_setup(self, skip_optional: bool = False) -> bool:
        """Run the complete setup process."""
        
        self._log("Starting AutoGen enhanced setup process...")
        
        success = True
        
        for step_id in self.execution_order:
            step = self.steps[step_id]
            
            # Skip optional steps if requested
            if skip_optional and step.optional:
                step.status = SetupStatus.SKIPPED
                self._log(f"Skipping optional step: {step.name}")
                continue
            
            # Check dependencies
            if not self._check_dependencies(step):
                step.status = SetupStatus.FAILED
                step.error_message = "Dependencies not met"
                success = False
                break
            
            # Execute step
            step_success = await self._execute_step(step)
            
            if not step_success:
                if not step.optional:
                    success = False
                    break
                else:
                    self._log(f"Optional step failed: {step.name}")
        
        if success:
            self._log("Setup completed successfully!")
            await self._create_setup_summary()
        else:
            self._log("Setup failed. Check the logs for details.")
        
        return success
    
    async def _execute_step(self, step: SetupStep) -> bool:
        """Execute a single setup step."""
        
        self._log(f"Executing: {step.name}")
        self._update_progress(step.id, SetupStatus.RUNNING)
        
        step.status = SetupStatus.RUNNING
        step.start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                step.function(),
                timeout=step.timeout
            )
            
            step.result = result
            step.status = SetupStatus.COMPLETED
            step.end_time = time.time()
            
            self._log(f"Completed: {step.name}")
            self._update_progress(step.id, SetupStatus.COMPLETED)
            
            return True
            
        except asyncio.TimeoutError:
            step.status = SetupStatus.FAILED
            step.error_message = f"Step timed out after {step.timeout} seconds"
            step.end_time = time.time()
            
            self._log(f"Failed (timeout): {step.name}")
            self._update_progress(step.id, SetupStatus.FAILED, step.error_message)
            
            return False
            
        except Exception as e:
            step.status = SetupStatus.FAILED
            step.error_message = str(e)
            step.end_time = time.time()
            
            self._log(f"Failed: {step.name} - {e}")
            self._update_progress(step.id, SetupStatus.FAILED, step.error_message)
            
            return False
    
    def _check_dependencies(self, step: SetupStep) -> bool:
        """Check if step dependencies are met."""
        
        for dep_id in step.dependencies:
            if dep_id not in self.steps:
                return False
            
            dep_step = self.steps[dep_id]
            if dep_step.status != SetupStatus.COMPLETED:
                return False
        
        return True
    
    def _update_execution_order(self) -> None:
        """Update execution order based on dependencies."""
        
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {step_id}")
            
            if step_id not in visited:
                temp_visited.add(step_id)
                
                step = self.steps.get(step_id)
                if step:
                    for dep_id in step.dependencies:
                        if dep_id in self.steps:
                            visit(dep_id)
                
                temp_visited.remove(step_id)
                visited.add(step_id)
                order.append(step_id)
        
        for step_id in self.steps:
            if step_id not in visited:
                visit(step_id)
        
        self.execution_order = order
    
    def _update_progress(self, step_id: str, status: SetupStatus, message: Optional[str] = None) -> None:
        """Update progress and call callback if set."""
        
        if self.progress_callback:
            self.progress_callback(step_id, status, message)
    
    def _log(self, message: str) -> None:
        """Log a message."""
        
        log_entry = {
            "timestamp": time.time(),
            "message": message
        }
        
        self.setup_log.append(log_entry)
        
        if self.verbose:
            print(f"[Setup] {message}")
    
    # Default setup step implementations
    
    async def _validate_environment(self) -> Dict[str, Any]:
        """Validate the environment."""
        
        validation_result = await self.environment_setup.validate_environment()
        
        if not validation_result["valid"]:
            raise Exception(f"Environment validation failed: {validation_result['errors']}")
        
        return validation_result
    
    async def _create_directories(self) -> Dict[str, Any]:
        """Create necessary directories."""
        
        directories = [
            "config",
            "plugins",
            "memory",
            "logs",
            "cache"
        ]
        
        created_dirs = []
        
        for dir_name in directories:
            dir_path = self.project_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(dir_path))
        
        return {"created_directories": created_dirs}
    
    async def _install_dependencies(self) -> Dict[str, Any]:
        """Install required dependencies."""
        
        # Core dependencies for enhanced AutoGen
        dependencies = [
            DependencyInfo("chromadb", ">=0.4.0", "Vector database for persistent memory"),
            DependencyInfo("sentence-transformers", ">=2.2.0", "Embeddings for semantic search"),
            DependencyInfo("psutil", ">=5.9.0", "System monitoring"),
            DependencyInfo("aiofiles", ">=23.0.0", "Async file operations"),
            DependencyInfo("pydantic", ">=2.0.0", "Data validation"),
            DependencyInfo("fastapi", ">=0.100.0", "Web framework for dashboard", optional=True),
            DependencyInfo("uvicorn", ">=0.23.0", "ASGI server", optional=True),
            DependencyInfo("websockets", ">=11.0.0", "WebSocket support", optional=True)
        ]
        
        installation_results = []
        
        for dep in dependencies:
            try:
                result = await self.dependency_manager.install_dependency(dep)
                installation_results.append({
                    "name": dep.name,
                    "success": result,
                    "optional": dep.optional
                })
            except Exception as e:
                if not dep.optional:
                    raise Exception(f"Failed to install required dependency {dep.name}: {e}")
                else:
                    installation_results.append({
                        "name": dep.name,
                        "success": False,
                        "optional": True,
                        "error": str(e)
                    })
        
        return {"installation_results": installation_results}
    
    async def _setup_vector_store(self) -> Dict[str, Any]:
        """Setup ChromaDB vector store."""
        
        try:
            import chromadb
            
            # Create vector store directory
            vector_store_path = self.project_path / "memory" / "vector_store"
            vector_store_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(path=str(vector_store_path))
            
            # Create default collection
            collection = client.get_or_create_collection(
                name="autogen_memory",
                metadata={"description": "AutoGen persistent memory"}
            )
            
            return {
                "vector_store_path": str(vector_store_path),
                "collection_name": "autogen_memory",
                "status": "initialized"
            }
            
        except ImportError:
            raise Exception("ChromaDB not available. Install with: pip install chromadb")
    
    async def _create_config(self) -> Dict[str, Any]:
        """Create configuration files."""
        
        # Create main configuration
        config_template = ConfigTemplate.get_default_template()
        config = self.config_manager.create_config_from_template(config_template)
        
        # Save configuration
        config_path = self.project_path / "config" / "autogen_config.json"
        self.config_manager.save_config(config, str(config_path))
        
        # Create plugin configuration
        plugin_config = {
            "plugins": {
                "logging_plugin": {"enabled": True, "priority": 10},
                "caching_plugin": {"enabled": True, "priority": 20},
                "analytics_plugin": {"enabled": False, "priority": 5}
            },
            "plugin_directories": ["./plugins"],
            "auto_load": True
        }
        
        plugin_config_path = self.project_path / "config" / "plugin_config.json"
        with open(plugin_config_path, 'w') as f:
            json.dump(plugin_config, f, indent=2)
        
        return {
            "main_config": str(config_path),
            "plugin_config": str(plugin_config_path)
        }
    
    async def _validate_config(self) -> Dict[str, Any]:
        """Validate configuration files."""
        
        config_path = self.project_path / "config" / "autogen_config.json"
        
        if not config_path.exists():
            raise Exception("Configuration file not found")
        
        config = self.config_manager.load_config(str(config_path))
        validation_result = self.config_manager.validate_config(config)
        
        if not validation_result["valid"]:
            raise Exception(f"Configuration validation failed: {validation_result['errors']}")
        
        return validation_result
    
    async def _initialize_plugins(self) -> Dict[str, Any]:
        """Initialize plugin system."""
        
        plugin_dir = self.project_path / "plugins"
        plugin_dir.mkdir(exist_ok=True)
        
        # Create example plugin
        example_plugin_content = '''"""
Example plugin for AutoGen.
"""

from autogen_ext.plugins import BasePlugin, PluginType, PluginCapability

class ExamplePlugin(BasePlugin):
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin demonstrating the plugin system"
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.EXTENSION
    
    @property
    def capabilities(self) -> set:
        return {PluginCapability.CUSTOM}
    
    async def _initialize(self) -> None:
        print(f"Initialized {self.name}")
'''
        
        example_plugin_path = plugin_dir / "example_plugin.py"
        with open(example_plugin_path, 'w') as f:
            f.write(example_plugin_content)
        
        return {
            "plugin_directory": str(plugin_dir),
            "example_plugin": str(example_plugin_path)
        }
    
    async def _run_tests(self) -> Dict[str, Any]:
        """Run basic functionality tests."""
        
        test_results = []
        
        # Test 1: Import core modules
        try:
            from autogen_ext.optimization import ContextCompressor
            from autogen_ext.monitoring import HealthMonitor
            from autogen_ext.execution import ParallelAgentExecutor
            test_results.append({"test": "import_modules", "success": True})
        except Exception as e:
            test_results.append({"test": "import_modules", "success": False, "error": str(e)})
        
        # Test 2: Create basic components
        try:
            compressor = ContextCompressor()
            monitor = HealthMonitor()
            test_results.append({"test": "create_components", "success": True})
        except Exception as e:
            test_results.append({"test": "create_components", "success": False, "error": str(e)})
        
        # Test 3: Configuration loading
        try:
            config_path = self.project_path / "config" / "autogen_config.json"
            config = self.config_manager.load_config(str(config_path))
            test_results.append({"test": "load_config", "success": True})
        except Exception as e:
            test_results.append({"test": "load_config", "success": False, "error": str(e)})
        
        return {"test_results": test_results}
    
    async def _create_setup_summary(self) -> None:
        """Create a setup summary file."""
        
        summary = {
            "setup_completed_at": time.time(),
            "project_path": str(self.project_path),
            "steps_executed": [],
            "configuration": {
                "main_config": str(self.project_path / "config" / "autogen_config.json"),
                "plugin_config": str(self.project_path / "config" / "plugin_config.json")
            },
            "next_steps": [
                "Review configuration files in the config/ directory",
                "Add your API keys to the configuration",
                "Explore example plugins in the plugins/ directory",
                "Run your first enhanced AutoGen agent",
                "Check the documentation for advanced features"
            ]
        }
        
        for step_id, step in self.steps.items():
            summary["steps_executed"].append({
                "id": step.id,
                "name": step.name,
                "status": step.status.value,
                "duration": (step.end_time - step.start_time) if step.start_time and step.end_time else None,
                "error": step.error_message
            })
        
        summary_path = self.project_path / "setup_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self._log(f"Setup summary saved to: {summary_path}")
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get current setup status."""
        
        status_counts = {}
        for status in SetupStatus:
            status_counts[status.value] = sum(
                1 for step in self.steps.values() if step.status == status
            )
        
        return {
            "total_steps": len(self.steps),
            "status_counts": status_counts,
            "execution_order": self.execution_order,
            "current_step": next(
                (step.name for step in self.steps.values() if step.status == SetupStatus.RUNNING),
                None
            )
        }
    
    def get_setup_log(self) -> List[Dict[str, Any]]:
        """Get setup log entries."""
        return self.setup_log.copy()
    
    async def cleanup_failed_setup(self) -> bool:
        """Clean up after a failed setup."""
        
        try:
            # Remove created directories if they're empty
            cleanup_dirs = ["config", "plugins", "memory", "logs", "cache"]
            
            for dir_name in cleanup_dirs:
                dir_path = self.project_path / dir_name
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
            
            # Remove setup summary if it exists
            summary_path = self.project_path / "setup_summary.json"
            if summary_path.exists():
                summary_path.unlink()
            
            return True
            
        except Exception as e:
            self._log(f"Cleanup failed: {e}")
            return False
