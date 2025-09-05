"""
Configuration builder for creating PolarisConfiguration instances.
"""

from typing import List, Union
from pathlib import Path

from .core import PolarisConfiguration
from .sources import ConfigurationSource, YAMLConfigurationSource, EnvironmentConfigurationSource


class ConfigurationBuilder:
    """
    Builder for creating PolarisConfiguration instances with multiple sources.
    
    Supports YAML files, environment variables, custom sources, validation, and hot-reloading.
    """
    
    def __init__(self):
        self._sources: List[ConfigurationSource] = []
        self._enable_hot_reload: bool = False
    
    def add_yaml_source(self, path: Union[str, Path], priority: int = 100) -> 'ConfigurationBuilder':
        """
        Add a YAML configuration source.
        
        Args:
            path: Path to the YAML configuration file
            priority: Priority of this source (higher = more important)
        """
        source = YAMLConfigurationSource(path, priority)
        self._sources.append(source)
        return self
    
    def add_environment_source(self, prefix: str = "POLARIS_", priority: int = 200, validate: bool = True) -> 'ConfigurationBuilder':
        """
        Add environment variable configuration source.
        
        Args:
            prefix: Environment variable prefix (default: POLARIS_)
            priority: Priority of this source (higher = more important)
            validate: Whether to validate environment variables
        """
        source = EnvironmentConfigurationSource(prefix, priority, validate)
        self._sources.append(source)
        return self
    
    def add_source(self, source: ConfigurationSource) -> 'ConfigurationBuilder':
        """Add a custom configuration source."""
        self._sources.append(source)
        return self
    
    def add_defaults(self) -> 'ConfigurationBuilder':
        """Add default configuration sources (environment variables with POLARIS_ prefix)."""
        return self.add_environment_source("POLARIS_", 200, True)
    
    def enable_hot_reload(self, enable: bool = True) -> 'ConfigurationBuilder':
        """
        Enable or disable hot-reloading of configuration files.
        
        Args:
            enable: Whether to enable hot-reloading
        """
        self._enable_hot_reload = enable
        return self
    
    def build(self) -> PolarisConfiguration:
        """
        Build the configuration instance with all added sources.
        
        Returns:
            PolarisConfiguration instance with all sources loaded and merged
        """
        if not self._sources:
            # Add default environment source if no sources specified
            self.add_defaults()
        
        return PolarisConfiguration(self._sources.copy(), self._enable_hot_reload)