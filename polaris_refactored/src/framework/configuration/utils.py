"""
Utility functions for common configuration patterns.
"""

from typing import Union
from pathlib import Path

from .builder import ConfigurationBuilder
from .core import PolarisConfiguration


def load_configuration_from_file(file_path: Union[str, Path], enable_hot_reload: bool = False) -> PolarisConfiguration:
    """
    Load configuration from a single YAML file with environment variable overrides.
    
    Args:
        file_path: Path to the YAML configuration file
        enable_hot_reload: Whether to enable hot-reloading of the configuration file
        
    Returns:
        PolarisConfiguration instance
    """
    return (ConfigurationBuilder()
            .add_yaml_source(file_path, 100)
            .add_environment_source("POLARIS_", 200, True)
            .enable_hot_reload(enable_hot_reload)
            .build())


def load_default_configuration(enable_validation: bool = True) -> PolarisConfiguration:
    """
    Load default configuration with environment variable support.
    
    Args:
        enable_validation: Whether to validate environment variables
        
    Returns:
        PolarisConfiguration instance with default settings
    """
    return (ConfigurationBuilder()
            .add_environment_source("POLARIS_", 200, enable_validation)
            .build())


def create_configuration_builder() -> ConfigurationBuilder:
    """
    Create a new configuration builder.
    
    Returns:
        ConfigurationBuilder instance
    """
    return ConfigurationBuilder()


def load_configuration_with_hot_reload(file_path: Union[str, Path]) -> PolarisConfiguration:
    """
    Load configuration from a YAML file with hot-reloading enabled.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        PolarisConfiguration instance with hot-reloading enabled
    """
    return load_configuration_from_file(file_path, enable_hot_reload=True)


def load_hot_reload_configuration(file_path: Union[str, Path]) -> PolarisConfiguration:
    """
    Load configuration from a YAML file with hot-reloading enabled.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        PolarisConfiguration instance with hot-reloading enabled
    """
    return load_configuration_from_file(file_path, enable_hot_reload=True)