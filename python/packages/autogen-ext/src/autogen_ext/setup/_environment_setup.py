"""
Environment setup and validation for AutoGen enhanced system.
"""

import asyncio
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess


@dataclass
class EnvironmentRequirement:
    """Represents an environment requirement."""
    name: str
    description: str
    check_function: str
    required: bool = True
    min_version: Optional[str] = None
    max_version: Optional[str] = None


class EnvironmentValidator:
    """Validates system environment for AutoGen enhanced features."""
    
    def __init__(self):
        self.requirements = self._get_default_requirements()
    
    def _get_default_requirements(self) -> List[EnvironmentRequirement]:
        """Get default environment requirements."""
        
        return [
            EnvironmentRequirement(
                name="python_version",
                description="Python 3.8 or higher",
                check_function="check_python_version",
                min_version="3.8.0"
            ),
            EnvironmentRequirement(
                name="pip",
                description="pip package manager",
                check_function="check_pip_available"
            ),
            EnvironmentRequirement(
                name="git",
                description="Git version control",
                check_function="check_git_available",
                required=False
            ),
            EnvironmentRequirement(
                name="docker",
                description="Docker for secure code execution",
                check_function="check_docker_available",
                required=False
            ),
            EnvironmentRequirement(
                name="disk_space",
                description="At least 1GB free disk space",
                check_function="check_disk_space"
            ),
            EnvironmentRequirement(
                name="memory",
                description="At least 2GB available memory",
                check_function="check_memory"
            )
        ]
    
    async def validate_all(self) -> Dict[str, Any]:
        """Validate all environment requirements."""
        
        results = []
        errors = []
        warnings = []
        
        for requirement in self.requirements:
            try:
                check_method = getattr(self, requirement.check_function)
                result = await check_method(requirement)
                
                results.append({
                    "name": requirement.name,
                    "description": requirement.description,
                    "passed": result["passed"],
                    "message": result["message"],
                    "required": requirement.required,
                    "details": result.get("details", {})
                })
                
                if not result["passed"]:
                    if requirement.required:
                        errors.append(f"{requirement.description}: {result['message']}")
                    else:
                        warnings.append(f"{requirement.description}: {result['message']}")
                        
            except Exception as e:
                error_msg = f"Failed to check {requirement.name}: {e}"
                results.append({
                    "name": requirement.name,
                    "description": requirement.description,
                    "passed": False,
                    "message": error_msg,
                    "required": requirement.required,
                    "details": {}
                })
                
                if requirement.required:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
        
        return {
            "valid": len(errors) == 0,
            "results": results,
            "errors": errors,
            "warnings": warnings,
            "system_info": self.get_system_info()
        }
    
    async def check_python_version(self, requirement: EnvironmentRequirement) -> Dict[str, Any]:
        """Check Python version."""
        
        current_version = platform.python_version()
        
        if requirement.min_version:
            min_parts = [int(x) for x in requirement.min_version.split('.')]
            current_parts = [int(x) for x in current_version.split('.')]
            
            meets_min = current_parts >= min_parts
        else:
            meets_min = True
        
        if requirement.max_version:
            max_parts = [int(x) for x in requirement.max_version.split('.')]
            meets_max = current_parts <= max_parts
        else:
            meets_max = True
        
        passed = meets_min and meets_max
        
        return {
            "passed": passed,
            "message": f"Python {current_version}" + ("" if passed else f" (requires >= {requirement.min_version})"),
            "details": {
                "current_version": current_version,
                "min_version": requirement.min_version,
                "max_version": requirement.max_version
            }
        }
    
    async def check_pip_available(self, requirement: EnvironmentRequirement) -> Dict[str, Any]:
        """Check if pip is available."""
        
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                pip_version = result.stdout.strip()
                return {
                    "passed": True,
                    "message": f"pip available: {pip_version}",
                    "details": {"version": pip_version}
                }
            else:
                return {
                    "passed": False,
                    "message": "pip not available or not working",
                    "details": {"error": result.stderr}
                }
                
        except Exception as e:
            return {
                "passed": False,
                "message": f"Failed to check pip: {e}",
                "details": {"error": str(e)}
            }
    
    async def check_git_available(self, requirement: EnvironmentRequirement) -> Dict[str, Any]:
        """Check if git is available."""
        
        try:
            git_path = shutil.which("git")
            if git_path:
                result = subprocess.run(["git", "--version"], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    git_version = result.stdout.strip()
                    return {
                        "passed": True,
                        "message": f"Git available: {git_version}",
                        "details": {"version": git_version, "path": git_path}
                    }
            
            return {
                "passed": False,
                "message": "Git not found in PATH",
                "details": {}
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Failed to check git: {e}",
                "details": {"error": str(e)}
            }
    
    async def check_docker_available(self, requirement: EnvironmentRequirement) -> Dict[str, Any]:
        """Check if Docker is available."""
        
        try:
            docker_path = shutil.which("docker")
            if docker_path:
                result = subprocess.run(["docker", "--version"], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    docker_version = result.stdout.strip()
                    
                    # Check if Docker daemon is running
                    daemon_result = subprocess.run(["docker", "info"], 
                                                 capture_output=True, text=True, timeout=10)
                    
                    daemon_running = daemon_result.returncode == 0
                    
                    return {
                        "passed": True,
                        "message": f"Docker available: {docker_version}" + 
                                 ("" if daemon_running else " (daemon not running)"),
                        "details": {
                            "version": docker_version,
                            "path": docker_path,
                            "daemon_running": daemon_running
                        }
                    }
            
            return {
                "passed": False,
                "message": "Docker not found in PATH",
                "details": {}
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Failed to check Docker: {e}",
                "details": {"error": str(e)}
            }
    
    async def check_disk_space(self, requirement: EnvironmentRequirement) -> Dict[str, Any]:
        """Check available disk space."""
        
        try:
            import psutil
            
            # Check disk space in current directory
            current_path = Path.cwd()
            disk_usage = psutil.disk_usage(str(current_path))
            
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            
            # Require at least 1GB free space
            min_free_gb = 1.0
            passed = free_gb >= min_free_gb
            
            return {
                "passed": passed,
                "message": f"{free_gb:.1f}GB free" + ("" if passed else f" (requires >= {min_free_gb}GB)"),
                "details": {
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "min_required_gb": min_free_gb
                }
            }
            
        except ImportError:
            return {
                "passed": False,
                "message": "psutil not available for disk space check",
                "details": {}
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Failed to check disk space: {e}",
                "details": {"error": str(e)}
            }
    
    async def check_memory(self, requirement: EnvironmentRequirement) -> Dict[str, Any]:
        """Check available memory."""
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            # Require at least 2GB available memory
            min_available_gb = 2.0
            passed = available_gb >= min_available_gb
            
            return {
                "passed": passed,
                "message": f"{available_gb:.1f}GB available" + 
                         ("" if passed else f" (requires >= {min_available_gb}GB)"),
                "details": {
                    "available_gb": available_gb,
                    "total_gb": total_gb,
                    "used_gb": (memory.used / (1024**3)),
                    "min_required_gb": min_available_gb,
                    "percent_used": memory.percent
                }
            }
            
        except ImportError:
            return {
                "passed": False,
                "message": "psutil not available for memory check",
                "details": {}
            }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Failed to check memory: {e}",
                "details": {"error": str(e)}
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "USER": os.environ.get("USER", "")
            }
        }


class EnvironmentSetup:
    """Sets up the environment for AutoGen enhanced features."""
    
    def __init__(self):
        self.validator = EnvironmentValidator()
    
    async def validate_environment(self) -> Dict[str, Any]:
        """Validate the current environment."""
        return await self.validator.validate_all()
    
    async def setup_python_environment(self, project_path: str) -> Dict[str, Any]:
        """Set up Python environment (virtual environment, etc.)."""
        
        project_dir = Path(project_path)
        venv_path = project_dir / "venv"
        
        setup_steps = []
        
        # Check if virtual environment already exists
        if venv_path.exists():
            setup_steps.append({
                "step": "check_venv",
                "message": "Virtual environment already exists",
                "success": True
            })
        else:
            # Create virtual environment
            try:
                result = subprocess.run([
                    sys.executable, "-m", "venv", str(venv_path)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    setup_steps.append({
                        "step": "create_venv",
                        "message": "Virtual environment created successfully",
                        "success": True
                    })
                else:
                    setup_steps.append({
                        "step": "create_venv",
                        "message": f"Failed to create virtual environment: {result.stderr}",
                        "success": False
                    })
                    
            except Exception as e:
                setup_steps.append({
                    "step": "create_venv",
                    "message": f"Failed to create virtual environment: {e}",
                    "success": False
                })
        
        # Create activation scripts
        activation_scripts = self._create_activation_scripts(project_dir, venv_path)
        setup_steps.extend(activation_scripts)
        
        return {
            "success": all(step["success"] for step in setup_steps),
            "steps": setup_steps,
            "venv_path": str(venv_path)
        }
    
    def _create_activation_scripts(self, project_dir: Path, venv_path: Path) -> List[Dict[str, Any]]:
        """Create convenient activation scripts."""
        
        scripts = []
        
        # Create activate script for Unix/Linux/Mac
        if platform.system() != "Windows":
            activate_script = project_dir / "activate.sh"
            try:
                with open(activate_script, 'w') as f:
                    f.write(f"""#!/bin/bash
# AutoGen Enhanced Environment Activation Script

echo "Activating AutoGen Enhanced environment..."

# Activate virtual environment
source {venv_path}/bin/activate

# Set environment variables
export AUTOGEN_PROJECT_ROOT="{project_dir}"
export AUTOGEN_CONFIG_PATH="{project_dir}/config"

echo "Environment activated!"
echo "Python: $(which python)"
echo "Project root: $AUTOGEN_PROJECT_ROOT"
""")
                
                # Make executable
                activate_script.chmod(0o755)
                
                scripts.append({
                    "step": "create_activate_script",
                    "message": f"Created activation script: {activate_script}",
                    "success": True
                })
                
            except Exception as e:
                scripts.append({
                    "step": "create_activate_script",
                    "message": f"Failed to create activation script: {e}",
                    "success": False
                })
        
        # Create activate script for Windows
        if platform.system() == "Windows":
            activate_script = project_dir / "activate.bat"
            try:
                with open(activate_script, 'w') as f:
                    f.write(f"""@echo off
REM AutoGen Enhanced Environment Activation Script

echo Activating AutoGen Enhanced environment...

REM Activate virtual environment
call {venv_path}\\Scripts\\activate.bat

REM Set environment variables
set AUTOGEN_PROJECT_ROOT={project_dir}
set AUTOGEN_CONFIG_PATH={project_dir}\\config

echo Environment activated!
echo Python: %VIRTUAL_ENV%\\Scripts\\python.exe
echo Project root: %AUTOGEN_PROJECT_ROOT%
""")
                
                scripts.append({
                    "step": "create_activate_script_windows",
                    "message": f"Created Windows activation script: {activate_script}",
                    "success": True
                })
                
            except Exception as e:
                scripts.append({
                    "step": "create_activate_script_windows",
                    "message": f"Failed to create Windows activation script: {e}",
                    "success": False
                })
        
        return scripts
    
    async def setup_directories(self, project_path: str) -> Dict[str, Any]:
        """Set up project directory structure."""
        
        project_dir = Path(project_path)
        
        directories = [
            "config",
            "plugins",
            "memory",
            "memory/vector_store",
            "logs",
            "cache",
            "data",
            "scripts",
            "tests"
        ]
        
        created_dirs = []
        errors = []
        
        for dir_name in directories:
            dir_path = project_dir / dir_name
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
            except Exception as e:
                errors.append(f"Failed to create {dir_path}: {e}")
        
        return {
            "success": len(errors) == 0,
            "created_directories": created_dirs,
            "errors": errors
        }
    
    async def setup_environment_files(self, project_path: str) -> Dict[str, Any]:
        """Set up environment configuration files."""
        
        project_dir = Path(project_path)
        created_files = []
        errors = []
        
        # Create .env template
        env_template_path = project_dir / ".env.template"
        try:
            with open(env_template_path, 'w') as f:
                f.write("""# AutoGen Enhanced Environment Variables
# Copy this file to .env and fill in your values

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure OpenAI Configuration (optional)
# AZURE_OPENAI_API_KEY=your_azure_api_key_here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2023-12-01-preview

# Other API Keys (optional)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# Database Configuration (optional)
# DATABASE_URL=postgresql://user:password@localhost/dbname

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/autogen.log

# Memory Configuration
MEMORY_STORAGE_PATH=./memory
VECTOR_STORE_PATH=./memory/vector_store

# Security Configuration
ENABLE_CODE_EXECUTION=true
USE_DOCKER_SANDBOX=true
""")
            
            created_files.append(str(env_template_path))
            
        except Exception as e:
            errors.append(f"Failed to create .env template: {e}")
        
        # Create .gitignore
        gitignore_path = project_dir / ".gitignore"
        try:
            with open(gitignore_path, 'w') as f:
                f.write("""# AutoGen Enhanced .gitignore

# Environment files
.env
.env.local
.env.*.local

# Virtual environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Memory and cache
memory/
cache/
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
""")
            
            created_files.append(str(gitignore_path))
            
        except Exception as e:
            errors.append(f"Failed to create .gitignore: {e}")
        
        return {
            "success": len(errors) == 0,
            "created_files": created_files,
            "errors": errors
        }
    
    def get_setup_instructions(self, project_path: str) -> str:
        """Get setup instructions for the user."""
        
        project_dir = Path(project_path)
        
        instructions = f"""
# AutoGen Enhanced Setup Instructions

Your AutoGen Enhanced project has been set up in: {project_dir}

## Next Steps:

1. **Activate the environment:**
   ```bash
   cd {project_dir}
   source activate.sh  # On Unix/Linux/Mac
   # OR
   activate.bat        # On Windows
   ```

2. **Configure your API keys:**
   - Copy `.env.template` to `.env`
   - Edit `.env` and add your API keys
   - At minimum, set your OPENAI_API_KEY

3. **Review configuration:**
   - Check `config/autogen_config.json` for main settings
   - Modify `config/plugin_config.json` for plugin settings

4. **Test the installation:**
   ```python
   from autogen_ext.setup import QuickStart
   
   # Run a quick test
   quick_start = QuickStart()
   await quick_start.run_basic_test()
   ```

5. **Explore features:**
   - Check the `plugins/` directory for example plugins
   - Review `memory/` for persistent memory storage
   - Look at `logs/` for system logs

## Directory Structure:
```
{project_dir.name}/
├── config/           # Configuration files
├── plugins/          # Custom plugins
├── memory/           # Persistent memory storage
├── logs/             # Log files
├── cache/            # Temporary cache
├── data/             # Data files
├── scripts/          # Utility scripts
├── tests/            # Test files
├── .env.template     # Environment variables template
├── .gitignore        # Git ignore rules
└── activate.sh       # Environment activation script
```

## Getting Help:
- Check the documentation in the `docs/` directory
- Run the built-in diagnostics: `python -m autogen_ext.setup.diagnostics`
- Visit the AutoGen Enhanced GitHub repository for examples

Happy coding with AutoGen Enhanced! 🚀
"""
        
        return instructions
