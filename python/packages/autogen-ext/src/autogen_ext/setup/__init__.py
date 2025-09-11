"""
Simplified setup and configuration system for AutoGen enhancements.

This module provides easy-to-use setup utilities for:
- Environment configuration
- Dependency management
- Service initialization
- Configuration validation
- Quick start templates
"""

from ._setup_manager import SetupManager, SetupStep, SetupStatus
from ._config_manager import ConfigManager, ConfigTemplate, ConfigValidator
from ._dependency_manager import DependencyManager, DependencyInfo, InstallationError
from ._environment_setup import EnvironmentSetup, EnvironmentValidator
from ._quick_start import QuickStart, ProjectTemplate

__all__ = [
    "SetupManager",
    "SetupStep", 
    "SetupStatus",
    "ConfigManager",
    "ConfigTemplate",
    "ConfigValidator",
    "DependencyManager",
    "DependencyInfo",
    "InstallationError",
    "EnvironmentSetup",
    "EnvironmentValidator", 
    "QuickStart",
    "ProjectTemplate"
]
