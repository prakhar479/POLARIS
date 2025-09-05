"""
Configuration sources for loading configuration data.
"""

import os
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
from pathlib import Path

from ...infrastructure.exceptions import ConfigurationError


class ConfigurationSource(ABC):
    """Abstract base class for configuration sources."""
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration data from the source."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get the priority of this source (higher number = higher priority)."""
        pass


class YAMLConfigurationSource(ConfigurationSource):
    """YAML file configuration source."""
    
    def __init__(self, file_path: Union[str, Path], priority: int = 100):
        self.file_path = Path(file_path)
        self.priority = priority
        self._last_modified = None
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.file_path}",
                "CONFIG_FILE_NOT_FOUND",
                {"file_path": str(self.file_path)}
            )
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                return data
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {self.file_path}",
                "INVALID_YAML",
                {"file_path": str(self.file_path), "yaml_error": str(e)}
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Error reading configuration file: {self.file_path}",
                "CONFIG_READ_ERROR",
                {"file_path": str(self.file_path), "error": str(e)}
            ) from e
    
    def get_priority(self) -> int:
        return self.priority
    
    def has_changed(self) -> bool:
        """Check if the file has been modified since last load."""
        if not self.file_path.exists():
            return False
        
        current_modified = self.file_path.stat().st_mtime
        if self._last_modified is None:
            self._last_modified = current_modified
            return False
        
        if current_modified != self._last_modified:
            self._last_modified = current_modified
            return True
        
        return False


class EnvironmentConfigurationSource(ConfigurationSource):
    """Environment variable configuration source."""
    
    def __init__(self, prefix: str = "POLARIS_", priority: int = 200, validate: bool = True):
        self.prefix = prefix.upper()
        self.priority = priority
        self.validate = validate
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(self.prefix):].lower()
                self._set_nested_value(config, config_key, self._parse_value(value, config_key))
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set a nested configuration value using underscore notation."""
        parts = key.split('_')
        current = config
        
        # Handle special cases for configuration structure
        # Convert FRAMEWORK_NATS_CONFIG_TIMEOUT to framework.nats_config.timeout
        if len(parts) >= 3 and parts[1] in ['nats', 'telemetry', 'logging'] and parts[2] == 'config':
            # Reconstruct as framework.{service}_config.{property}
            service_config_key = f"{parts[1]}_config"
            if parts[0] not in current:
                current[parts[0]] = {}
            if service_config_key not in current[parts[0]]:
                current[parts[0]][service_config_key] = {}
            
            # Set the property in the service config
            property_path = parts[3:]
            target = current[parts[0]][service_config_key]
            for part in property_path[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            if property_path:
                target[property_path[-1]] = value
            else:
                # If no property path, set the value directly
                current[parts[0]][service_config_key] = value
        else:
            # Standard nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    # Convert to dict if it's not already
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
    
    def _parse_value(self, value: str, key: str = "") -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle known list fields - even single values should be converted to lists
        list_fields = {
            'framework_nats_config_servers',
            'framework_plugin_search_paths'
        }
        
        if key.lower() in list_fields:
            # Always return as list for these fields
            if ',' in value:
                return [item.strip() for item in value.split(',')]
            else:
                return [value.strip()]
        
        # Try to parse as list (comma-separated) for other fields
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    def get_priority(self) -> int:
        return self.priority