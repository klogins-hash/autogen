"""
Dependency management for AutoGen enhanced setup.
"""

import asyncio
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import importlib
import pkg_resources
from pathlib import Path


class InstallationError(Exception):
    """Exception raised when dependency installation fails."""
    pass


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version_spec: str = ""
    description: str = ""
    optional: bool = False
    install_command: Optional[str] = None
    post_install_check: Optional[str] = None


class DependencyManager:
    """Manages installation and validation of dependencies."""
    
    def __init__(self):
        self.installed_packages: Dict[str, str] = {}
        self._refresh_installed_packages()
    
    def _refresh_installed_packages(self) -> None:
        """Refresh the list of installed packages."""
        
        try:
            installed_packages = pkg_resources.working_set
            self.installed_packages = {
                pkg.project_name.lower(): pkg.version 
                for pkg in installed_packages
            }
        except Exception:
            # Fallback to empty dict if pkg_resources fails
            self.installed_packages = {}
    
    async def check_dependency(self, dependency: DependencyInfo) -> Dict[str, Any]:
        """Check if a dependency is installed and meets version requirements."""
        
        package_name = dependency.name.lower()
        
        # Check if package is installed
        if package_name not in self.installed_packages:
            return {
                "installed": False,
                "version": None,
                "meets_requirements": False,
                "message": f"Package {dependency.name} is not installed"
            }
        
        installed_version = self.installed_packages[package_name]
        
        # Check version requirements if specified
        meets_requirements = True
        if dependency.version_spec:
            try:
                requirement = pkg_resources.Requirement.parse(f"{dependency.name}{dependency.version_spec}")
                meets_requirements = installed_version in requirement
            except Exception:
                meets_requirements = False
        
        return {
            "installed": True,
            "version": installed_version,
            "meets_requirements": meets_requirements,
            "message": f"Package {dependency.name} version {installed_version} is installed"
        }
    
    async def install_dependency(self, dependency: DependencyInfo) -> bool:
        """Install a dependency."""
        
        # Check if already installed and meets requirements
        check_result = await self.check_dependency(dependency)
        if check_result["installed"] and check_result["meets_requirements"]:
            return True
        
        # Determine install command
        if dependency.install_command:
            install_cmd = dependency.install_command.split()
        else:
            package_spec = f"{dependency.name}{dependency.version_spec}" if dependency.version_spec else dependency.name
            install_cmd = [sys.executable, "-m", "pip", "install", package_spec]
        
        try:
            # Run installation
            process = await asyncio.create_subprocess_exec(
                *install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown installation error"
                raise InstallationError(f"Failed to install {dependency.name}: {error_msg}")
            
            # Refresh installed packages
            self._refresh_installed_packages()
            
            # Verify installation
            check_result = await self.check_dependency(dependency)
            if not check_result["installed"]:
                raise InstallationError(f"Package {dependency.name} was not found after installation")
            
            # Run post-install check if specified
            if dependency.post_install_check:
                await self._run_post_install_check(dependency)
            
            return True
            
        except Exception as e:
            if dependency.optional:
                return False
            else:
                raise InstallationError(f"Failed to install required dependency {dependency.name}: {e}")
    
    async def install_dependencies(self, dependencies: List[DependencyInfo]) -> Dict[str, Any]:
        """Install multiple dependencies."""
        
        results = []
        failed_required = []
        
        for dependency in dependencies:
            try:
                success = await self.install_dependency(dependency)
                results.append({
                    "name": dependency.name,
                    "success": success,
                    "optional": dependency.optional,
                    "description": dependency.description
                })
                
                if not success and not dependency.optional:
                    failed_required.append(dependency.name)
                    
            except Exception as e:
                results.append({
                    "name": dependency.name,
                    "success": False,
                    "optional": dependency.optional,
                    "error": str(e),
                    "description": dependency.description
                })
                
                if not dependency.optional:
                    failed_required.append(dependency.name)
        
        return {
            "results": results,
            "success": len(failed_required) == 0,
            "failed_required": failed_required
        }
    
    async def check_dependencies(self, dependencies: List[DependencyInfo]) -> Dict[str, Any]:
        """Check multiple dependencies without installing."""
        
        results = []
        missing_required = []
        
        for dependency in dependencies:
            check_result = await self.check_dependency(dependency)
            results.append({
                "name": dependency.name,
                "installed": check_result["installed"],
                "version": check_result["version"],
                "meets_requirements": check_result["meets_requirements"],
                "optional": dependency.optional,
                "message": check_result["message"]
            })
            
            if not check_result["installed"] and not dependency.optional:
                missing_required.append(dependency.name)
            elif check_result["installed"] and not check_result["meets_requirements"] and not dependency.optional:
                missing_required.append(f"{dependency.name} (version mismatch)")
        
        return {
            "results": results,
            "all_satisfied": len(missing_required) == 0,
            "missing_required": missing_required
        }
    
    async def _run_post_install_check(self, dependency: DependencyInfo) -> None:
        """Run post-installation check for a dependency."""
        
        if not dependency.post_install_check:
            return
        
        try:
            # Try to import the module to verify it's working
            if dependency.post_install_check.startswith("import "):
                module_name = dependency.post_install_check[7:]
                importlib.import_module(module_name)
            else:
                # Run as Python code
                exec(dependency.post_install_check)
                
        except Exception as e:
            raise InstallationError(f"Post-install check failed for {dependency.name}: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information relevant to dependency management."""
        
        return {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform,
            "installed_packages_count": len(self.installed_packages),
            "pip_version": self._get_pip_version()
        }
    
    def _get_pip_version(self) -> Optional[str]:
        """Get pip version."""
        
        try:
            import pip
            return pip.__version__
        except (ImportError, AttributeError):
            return None
    
    def create_requirements_file(self, dependencies: List[DependencyInfo], file_path: str) -> None:
        """Create a requirements.txt file from dependencies."""
        
        requirements_path = Path(file_path)
        requirements_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(requirements_path, 'w') as f:
            f.write("# AutoGen Enhanced Dependencies\n")
            f.write("# Generated automatically - do not edit manually\n\n")
            
            # Required dependencies
            f.write("# Required dependencies\n")
            for dep in dependencies:
                if not dep.optional:
                    package_spec = f"{dep.name}{dep.version_spec}" if dep.version_spec else dep.name
                    comment = f"  # {dep.description}" if dep.description else ""
                    f.write(f"{package_spec}{comment}\n")
            
            f.write("\n# Optional dependencies\n")
            for dep in dependencies:
                if dep.optional:
                    package_spec = f"{dep.name}{dep.version_spec}" if dep.version_spec else dep.name
                    comment = f"  # {dep.description}" if dep.description else ""
                    f.write(f"# {package_spec}{comment}\n")
    
    async def install_from_requirements(self, requirements_file: str) -> bool:
        """Install dependencies from a requirements file."""
        
        requirements_path = Path(requirements_file)
        if not requirements_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {requirements_file}")
        
        try:
            install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
            
            process = await asyncio.create_subprocess_exec(
                *install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown installation error"
                raise InstallationError(f"Failed to install from requirements: {error_msg}")
            
            # Refresh installed packages
            self._refresh_installed_packages()
            return True
            
        except Exception as e:
            raise InstallationError(f"Failed to install from requirements file: {e}")
    
    def get_dependency_tree(self, package_name: str) -> Dict[str, Any]:
        """Get dependency tree for a package."""
        
        try:
            import pipdeptree
            
            # Get all packages
            packages = pipdeptree.get_installed_distributions()
            
            # Find the target package
            target_package = None
            for pkg in packages:
                if pkg.project_name.lower() == package_name.lower():
                    target_package = pkg
                    break
            
            if not target_package:
                return {"error": f"Package {package_name} not found"}
            
            # Get dependencies
            deps = pipdeptree.get_dependencies(target_package)
            
            def format_deps(deps_list):
                result = []
                for dep in deps_list:
                    result.append({
                        "name": dep.project_name,
                        "version": dep.version,
                        "dependencies": format_deps(dep.dependencies) if hasattr(dep, 'dependencies') else []
                    })
                return result
            
            return {
                "package": target_package.project_name,
                "version": target_package.version,
                "dependencies": format_deps(deps)
            }
            
        except ImportError:
            return {"error": "pipdeptree not available. Install with: pip install pipdeptree"}
        except Exception as e:
            return {"error": f"Failed to get dependency tree: {e}"}
    
    async def upgrade_dependency(self, dependency_name: str) -> bool:
        """Upgrade a dependency to the latest version."""
        
        try:
            upgrade_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", dependency_name]
            
            process = await asyncio.create_subprocess_exec(
                *upgrade_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown upgrade error"
                raise InstallationError(f"Failed to upgrade {dependency_name}: {error_msg}")
            
            # Refresh installed packages
            self._refresh_installed_packages()
            return True
            
        except Exception as e:
            raise InstallationError(f"Failed to upgrade {dependency_name}: {e}")
    
    def export_installed_packages(self, file_path: str) -> None:
        """Export currently installed packages to a requirements file."""
        
        requirements_path = Path(file_path)
        requirements_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(requirements_path, 'w') as f:
            f.write("# Exported installed packages\n")
            f.write(f"# Generated on {asyncio.get_event_loop().time()}\n\n")
            
            for package_name, version in sorted(self.installed_packages.items()):
                f.write(f"{package_name}=={version}\n")
    
    def get_outdated_packages(self) -> List[Dict[str, str]]:
        """Get list of outdated packages (requires pip-outdated or similar)."""
        
        try:
            # This is a simplified version - in practice you'd use pip list --outdated
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                import json
                outdated_packages = json.loads(result.stdout)
                return outdated_packages
            else:
                return []
                
        except Exception:
            return []
