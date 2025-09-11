"""
Configuration management for AutoGen enhanced setup.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import jsonschema
from enum import Enum


class ConfigSection(Enum):
    """Configuration sections."""
    GENERAL = "general"
    MODELS = "models"
    AGENTS = "agents"
    MEMORY = "memory"
    MONITORING = "monitoring"
    PLUGINS = "plugins"
    SECURITY = "security"


@dataclass
class ConfigTemplate:
    """Template for generating configurations."""
    name: str
    description: str
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    
    @staticmethod
    def get_default_template() -> 'ConfigTemplate':
        """Get the default configuration template."""
        
        return ConfigTemplate(
            name="default_autogen_config",
            description="Default configuration for enhanced AutoGen",
            sections={
                "general": {
                    "project_name": "AutoGen Enhanced Project",
                    "version": "1.0.0",
                    "environment": "development",
                    "log_level": "INFO",
                    "max_workers": 4
                },
                "models": {
                    "default_model": "gpt-4",
                    "model_configs": {
                        "gpt-4": {
                            "api_key": "${OPENAI_API_KEY}",
                            "base_url": "https://api.openai.com/v1",
                            "max_tokens": 4096,
                            "temperature": 0.7
                        },
                        "gpt-3.5-turbo": {
                            "api_key": "${OPENAI_API_KEY}",
                            "base_url": "https://api.openai.com/v1",
                            "max_tokens": 2048,
                            "temperature": 0.7
                        }
                    }
                },
                "agents": {
                    "orchestrator": {
                        "model": "gpt-4",
                        "system_message": "You are an orchestrator agent coordinating multiple specialized agents.",
                        "max_consecutive_auto_reply": 10
                    },
                    "web_surfer": {
                        "model": "gpt-4",
                        "browser_config": {
                            "headless": True,
                            "timeout": 30
                        }
                    },
                    "coder": {
                        "model": "gpt-4",
                        "execution_config": {
                            "use_docker": True,
                            "timeout": 60
                        }
                    }
                },
                "memory": {
                    "persistent_memory": {
                        "enabled": True,
                        "storage_path": "./memory",
                        "vector_store": {
                            "provider": "chromadb",
                            "collection_name": "autogen_memory",
                            "embedding_model": "all-MiniLM-L6-v2"
                        }
                    },
                    "context_compression": {
                        "enabled": True,
                        "strategy": "hybrid",
                        "max_tokens": 8000
                    }
                },
                "monitoring": {
                    "health_checks": {
                        "enabled": True,
                        "interval": 60,
                        "checks": ["memory", "disk", "api_connectivity"]
                    },
                    "metrics": {
                        "enabled": True,
                        "collection_interval": 30
                    },
                    "error_recovery": {
                        "enabled": True,
                        "max_retries": 3,
                        "backoff_factor": 2.0
                    }
                },
                "plugins": {
                    "enabled": True,
                    "directories": ["./plugins"],
                    "auto_load": True,
                    "default_plugins": {
                        "logging": {"enabled": True, "priority": 10},
                        "caching": {"enabled": True, "priority": 20}
                    }
                },
                "security": {
                    "content_filtering": {
                        "enabled": True,
                        "blocked_patterns": ["password", "api_key", "secret"]
                    },
                    "code_execution": {
                        "sandbox": True,
                        "allowed_imports": ["os", "sys", "json", "requests"],
                        "timeout": 30
                    }
                }
            },
            required_fields=[
                "general.project_name",
                "models.default_model",
                "models.model_configs"
            ],
            optional_fields=[
                "memory.persistent_memory.enabled",
                "monitoring.health_checks.enabled",
                "plugins.enabled"
            ]
        )
    
    @staticmethod
    def get_minimal_template() -> 'ConfigTemplate':
        """Get a minimal configuration template."""
        
        return ConfigTemplate(
            name="minimal_autogen_config",
            description="Minimal configuration for AutoGen",
            sections={
                "general": {
                    "project_name": "AutoGen Project",
                    "log_level": "INFO"
                },
                "models": {
                    "default_model": "gpt-3.5-turbo",
                    "model_configs": {
                        "gpt-3.5-turbo": {
                            "api_key": "${OPENAI_API_KEY}",
                            "max_tokens": 2048
                        }
                    }
                }
            },
            required_fields=[
                "models.default_model",
                "models.model_configs"
            ]
        )


class ConfigValidator:
    """Validates configuration files."""
    
    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """Get JSON schema for configuration validation."""
        
        return {
            "type": "object",
            "properties": {
                "general": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "version": {"type": "string"},
                        "environment": {"type": "string", "enum": ["development", "staging", "production"]},
                        "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                        "max_workers": {"type": "integer", "minimum": 1}
                    },
                    "required": ["project_name"]
                },
                "models": {
                    "type": "object",
                    "properties": {
                        "default_model": {"type": "string"},
                        "model_configs": {
                            "type": "object",
                            "patternProperties": {
                                ".*": {
                                    "type": "object",
                                    "properties": {
                                        "api_key": {"type": "string"},
                                        "base_url": {"type": "string"},
                                        "max_tokens": {"type": "integer", "minimum": 1},
                                        "temperature": {"type": "number", "minimum": 0, "maximum": 2}
                                    },
                                    "required": ["api_key"]
                                }
                            }
                        }
                    },
                    "required": ["default_model", "model_configs"]
                },
                "memory": {
                    "type": "object",
                    "properties": {
                        "persistent_memory": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "storage_path": {"type": "string"},
                                "vector_store": {
                                    "type": "object",
                                    "properties": {
                                        "provider": {"type": "string"},
                                        "collection_name": {"type": "string"},
                                        "embedding_model": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["general", "models"]
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration dictionary."""
        
        schema = ConfigValidator.get_config_schema()
        errors = []
        warnings = []
        
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        
        # Additional custom validations
        
        # Check model configuration
        if "models" in config:
            default_model = config["models"].get("default_model")
            model_configs = config["models"].get("model_configs", {})
            
            if default_model and default_model not in model_configs:
                errors.append(f"Default model '{default_model}' not found in model_configs")
            
            # Check for API keys
            for model_name, model_config in model_configs.items():
                api_key = model_config.get("api_key", "")
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    if not os.getenv(env_var):
                        warnings.append(f"Environment variable {env_var} not set for model {model_name}")
        
        # Check paths
        if "memory" in config and "persistent_memory" in config["memory"]:
            storage_path = config["memory"]["persistent_memory"].get("storage_path")
            if storage_path:
                path_obj = Path(storage_path)
                if not path_obj.parent.exists():
                    warnings.append(f"Parent directory for storage_path does not exist: {storage_path}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


class ConfigManager:
    """Manages configuration files and templates."""
    
    def __init__(self):
        self.templates: Dict[str, ConfigTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default configuration templates."""
        
        default_template = ConfigTemplate.get_default_template()
        minimal_template = ConfigTemplate.get_minimal_template()
        
        self.templates[default_template.name] = default_template
        self.templates[minimal_template.name] = minimal_template
    
    def add_template(self, template: ConfigTemplate) -> None:
        """Add a configuration template."""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a configuration template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())
    
    def create_config_from_template(
        self,
        template: ConfigTemplate,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a configuration from a template."""
        
        config = {}
        
        # Copy template sections
        for section_name, section_data in template.sections.items():
            config[section_name] = self._deep_copy_dict(section_data)
        
        # Apply overrides
        if overrides:
            config = self._merge_configs(config, overrides)
        
        return config
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a file."""
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to a file."""
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Expand environment variables in the config before saving
        expanded_config = self._expand_environment_variables(config)
        
        with open(config_file, 'w') as f:
            if config_file.suffix.lower() == '.json':
                json.dump(expanded_config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration."""
        return ConfigValidator.validate_config(config)
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations."""
        return self._merge_configs(base_config, override_config)
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_config_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def create_environment_file(self, config: Dict[str, Any], env_file_path: str) -> None:
        """Create a .env file with required environment variables."""
        
        env_vars = set()
        
        # Extract environment variable references
        self._extract_env_vars(config, env_vars)
        
        env_file = Path(env_file_path)
        
        with open(env_file, 'w') as f:
            f.write("# Environment variables for AutoGen Enhanced\n")
            f.write("# Copy this file to .env and fill in your values\n\n")
            
            for var in sorted(env_vars):
                f.write(f"{var}=your_value_here\n")
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a dictionary."""
        
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_dict(value)
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        
        return result
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        
        result = self._deep_copy_dict(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _expand_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand environment variables in configuration values."""
        
        def expand_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)  # Return original if env var not found
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            else:
                return value
        
        return expand_value(config)
    
    def _extract_env_vars(self, obj: Any, env_vars: set) -> None:
        """Extract environment variable references from configuration."""
        
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            env_vars.add(env_var)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._extract_env_vars(value, env_vars)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_env_vars(item, env_vars)
    
    def generate_config_documentation(self, template: ConfigTemplate) -> str:
        """Generate documentation for a configuration template."""
        
        doc = f"# {template.name}\n\n"
        doc += f"{template.description}\n\n"
        
        doc += "## Configuration Sections\n\n"
        
        for section_name, section_data in template.sections.items():
            doc += f"### {section_name}\n\n"
            doc += self._document_section(section_data, level=0)
            doc += "\n"
        
        if template.required_fields:
            doc += "## Required Fields\n\n"
            for field in template.required_fields:
                doc += f"- `{field}`\n"
            doc += "\n"
        
        if template.optional_fields:
            doc += "## Optional Fields\n\n"
            for field in template.optional_fields:
                doc += f"- `{field}`\n"
            doc += "\n"
        
        return doc
    
    def _document_section(self, section: Dict[str, Any], level: int = 0) -> str:
        """Generate documentation for a configuration section."""
        
        doc = ""
        indent = "  " * level
        
        for key, value in section.items():
            if isinstance(value, dict):
                doc += f"{indent}- `{key}`: Configuration object\n"
                doc += self._document_section(value, level + 1)
            else:
                value_type = type(value).__name__
                doc += f"{indent}- `{key}` ({value_type}): Default value `{value}`\n"
        
        return doc
