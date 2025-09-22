"""
Configuration Manager for SWIM POLARIS Adaptation System

Provides hierarchical configuration loading with environment variable overrides,
schema validation, and hot-reloading capabilities.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import jsonschema
from jsonschema import validate, ValidationError


@dataclass
class ConfigurationSource:
    """Represents a configuration source with priority."""
    name: str
    path: str
    priority: int  # Higher number = higher priority
    format: str  # 'yaml' or 'json'
    required: bool = True


class ConfigurationSchema:
    """JSON Schema for SWIM POLARIS configuration validation."""
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "framework": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "version"]
            },
            "message_bus": {
                "type": "object",
                "properties": {
                    "nats": {
                        "type": "object",
                        "properties": {
                            "servers": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "name": {"type": "string"},
                            "connection_timeout": {"type": "number"},
                            "reconnect_attempts": {"type": "integer"},
                            "reconnect_delay": {"type": "number"}
                        },
                        "required": ["servers", "name"]
                    }
                },
                "required": ["nats"]
            },
            "managed_systems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "connector_type": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "config": {"type": "object"}
                    },
                    "required": ["system_id", "connector_type"]
                }
            },
            "adaptation": {
                "type": "object",
                "properties": {
                    "strategies": {"type": "object"},
                    "thresholds": {"type": "object"},
                    "constraints": {"type": "object"},
                    "actions": {"type": "object"}
                }
            },
            "ablation": {
                "type": "object",
                "properties": {
                    "components": {"type": "object"},
                    "study_parameters": {"type": "object"}
                }
            }
        },
        "required": ["framework", "message_bus", "managed_systems"]
    }


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches configuration files for changes."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            self.logger.info(f"Configuration file changed: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_configuration())


class HierarchicalConfigurationManager:
    """
    Manages hierarchical configuration loading with environment overrides.
    
    Supports:
    - Multiple configuration sources with priorities
    - Environment variable overrides
    - Schema validation
    - Hot-reloading
    - Configuration merging
    """
    
    def __init__(self, base_config_dir: str):
        """Initialize the configuration manager.
        
        Args:
            base_config_dir: Base directory containing configuration files
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_config_dir = Path(base_config_dir)
        
        # Configuration state
        self.sources: List[ConfigurationSource] = []
        self.current_config: Dict[str, Any] = {}
        self.environment_prefix = "SWIM_POLARIS_"
        
        # Hot-reloading
        self.hot_reload_enabled = False
        self.file_observer: Optional[Observer] = None
        self.reload_callbacks: List[callable] = []
        
        # Schema validation
        self.schema_validator = ConfigurationSchema()
        self.validation_enabled = True
        
        # Initialize default sources
        self._setup_default_sources()
    
    def _setup_default_sources(self) -> None:
        """Setup default configuration sources."""
        # Base configuration (lowest priority)
        base_config = self.base_config_dir / "base_config.yaml"
        if base_config.exists():
            self.add_source("base", str(base_config), priority=1, format="yaml")
        
        # SWIM-specific configuration
        swim_config = self.base_config_dir / "swim_config.yaml"
        if swim_config.exists():
            self.add_source("swim", str(swim_config), priority=2, format="yaml")
        
        # Environment-specific configuration
        env = os.getenv("ENVIRONMENT", "development")
        env_config = self.base_config_dir / f"{env}_config.yaml"
        if env_config.exists():
            self.add_source("environment", str(env_config), priority=3, format="yaml")
        
        # Local overrides (highest priority)
        local_config = self.base_config_dir / "local_config.yaml"
        if local_config.exists():
            self.add_source("local", str(local_config), priority=4, format="yaml", required=False)
    
    def add_source(self, name: str, path: str, priority: int, format: str = "yaml", required: bool = True) -> None:
        """Add a configuration source.
        
        Args:
            name: Source name
            path: Path to configuration file
            priority: Priority (higher = more important)
            format: File format ('yaml' or 'json')
            required: Whether the file must exist
        """
        source = ConfigurationSource(
            name=name,
            path=path,
            priority=priority,
            format=format,
            required=required
        )
        
        self.sources.append(source)
        self.sources.sort(key=lambda x: x.priority)  # Sort by priority
        
        self.logger.info(f"Added configuration source: {name} (priority: {priority})")
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load and merge configuration from all sources.
        
        Returns:
            Merged configuration dictionary
        """
        merged_config = {}
        
        # Load and merge configurations in priority order
        for source in self.sources:
            try:
                config_data = self._load_config_file(source)
                if config_data:
                    merged_config = self._deep_merge(merged_config, config_data)
                    self.logger.debug(f"Loaded configuration from {source.name}")
                
            except Exception as e:
                if source.required:
                    self.logger.error(f"Failed to load required config {source.name}: {e}")
                    raise
                else:
                    self.logger.warning(f"Failed to load optional config {source.name}: {e}")
        
        # Apply environment variable overrides
        merged_config = self._apply_environment_overrides(merged_config)
        
        # Validate configuration
        if self.validation_enabled:
            self._validate_configuration(merged_config)
        
        self.current_config = merged_config
        self.logger.info("Configuration loaded successfully")
        
        return merged_config
    
    def _load_config_file(self, source: ConfigurationSource) -> Optional[Dict[str, Any]]:
        """Load configuration from a single file."""
        config_path = Path(source.path)
        
        if not config_path.exists():
            if source.required:
                raise FileNotFoundError(f"Required configuration file not found: {source.path}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                if source.format.lower() == 'yaml':
                    return yaml.safe_load(f)
                elif source.format.lower() == 'json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {source.format}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to parse {source.format} file {source.path}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_overrides = {}
        
        # Collect environment variables with our prefix
        for key, value in os.environ.items():
            if key.startswith(self.environment_prefix):
                # Convert environment variable to config path
                config_key = key[len(self.environment_prefix):].lower()
                config_path = config_key.split('_')
                
                # Parse value (try JSON first, then string)
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value
                
                # Set nested value
                self._set_nested_value(env_overrides, config_path, parsed_value)
        
        if env_overrides:
            self.logger.info(f"Applied {len(env_overrides)} environment overrides")
            config = self._deep_merge(config, env_overrides)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any) -> None:
        """Set a nested value in configuration dictionary."""
        current = config
        
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[path[-1]] = value
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema."""
        try:
            validate(instance=config, schema=self.schema_validator.SCHEMA)
            self.logger.debug("Configuration validation passed")
        
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e.message}")
            raise ValueError(f"Invalid configuration: {e.message}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        if not self.current_config:
            return self.load_configuration()
        return self.current_config
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'adaptation.thresholds.response_time')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_config()
        keys = path.split('.')
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_value(self, path: str, value: Any) -> None:
        """Set a configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path
            value: Value to set
        """
        config = self.get_config()
        keys = path.split('.')
        
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self.logger.debug(f"Set configuration value: {path} = {value}")
    
    def enable_hot_reload(self) -> None:
        """Enable hot-reloading of configuration files."""
        if self.hot_reload_enabled:
            return
        
        self.file_observer = Observer()
        event_handler = ConfigFileWatcher(self)
        
        # Watch all source directories
        watched_dirs = set()
        for source in self.sources:
            source_dir = Path(source.path).parent
            if source_dir not in watched_dirs:
                self.file_observer.schedule(event_handler, str(source_dir), recursive=False)
                watched_dirs.add(source_dir)
        
        self.file_observer.start()
        self.hot_reload_enabled = True
        
        self.logger.info("Hot-reload enabled for configuration files")
    
    def disable_hot_reload(self) -> None:
        """Disable hot-reloading."""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
        
        self.hot_reload_enabled = False
        self.logger.info("Hot-reload disabled")
    
    async def reload_configuration(self) -> None:
        """Reload configuration from all sources."""
        try:
            old_config = self.current_config.copy()
            new_config = self.load_configuration()
            
            # Notify callbacks of configuration change
            for callback in self.reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_config, new_config)
                    else:
                        callback(old_config, new_config)
                except Exception as e:
                    self.logger.error(f"Configuration reload callback failed: {e}")
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
    
    def add_reload_callback(self, callback: callable) -> None:
        """Add a callback to be called when configuration is reloaded.
        
        Args:
            callback: Function to call with (old_config, new_config) parameters
        """
        self.reload_callbacks.append(callback)
    
    def export_config(self, output_path: str, format: str = "yaml") -> None:
        """Export current configuration to file.
        
        Args:
            output_path: Path to output file
            format: Output format ('yaml' or 'json')
        """
        config = self.get_config()
        
        with open(output_path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Configuration exported to {output_path}")
    
    def get_ablation_config(self, ablation_name: str) -> Dict[str, Any]:
        """Get configuration for a specific ablation study.
        
        Args:
            ablation_name: Name of the ablation configuration
            
        Returns:
            Merged configuration with ablation overrides
        """
        # Load base configuration
        base_config = self.get_config()
        
        # Load ablation-specific configuration
        ablation_config_path = self.base_config_dir / "ablation_configs" / f"{ablation_name}.yaml"
        
        if not ablation_config_path.exists():
            raise FileNotFoundError(f"Ablation configuration not found: {ablation_config_path}")
        
        with open(ablation_config_path, 'r') as f:
            ablation_overrides = yaml.safe_load(f)
        
        # Merge configurations
        merged_config = self._deep_merge(base_config, ablation_overrides)
        
        # Validate merged configuration
        if self.validation_enabled:
            self._validate_configuration(merged_config)
        
        return merged_config
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.disable_hot_reload()
        self.reload_callbacks.clear()