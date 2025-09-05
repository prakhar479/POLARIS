"""
Configuration Management System

Type-safe configuration management with hierarchical configuration support,
YAML and environment variable sources, and comprehensive validation.
"""

from .models import (
    NATSConfiguration,
    TelemetryConfiguration,
    LoggingConfiguration,
    FrameworkConfiguration,
    ManagedSystemConfiguration
)

from .sources import (
    ConfigurationSource,
    YAMLConfigurationSource,
    EnvironmentConfigurationSource
)

from .validation import (
    ConfigurationValidator,
    ConfigurationValidationError
)

from .core import PolarisConfiguration

from .builder import ConfigurationBuilder

from .utils import (
    load_configuration_from_file,
    load_default_configuration,
    create_configuration_builder,
    load_configuration_with_hot_reload,
    load_hot_reload_configuration
)

__all__ = [
    # Models
    'NATSConfiguration',
    'TelemetryConfiguration', 
    'LoggingConfiguration',
    'FrameworkConfiguration',
    'ManagedSystemConfiguration',
    
    # Sources
    'ConfigurationSource',
    'YAMLConfigurationSource',
    'EnvironmentConfigurationSource',
    
    # Validation
    'ConfigurationValidator',
    'ConfigurationValidationError',
    
    # Core
    'PolarisConfiguration',
    
    # Builder
    'ConfigurationBuilder',
    
    # Utilities
    'load_configuration_from_file',
    'load_default_configuration',
    'create_configuration_builder',
    'load_configuration_with_hot_reload',
    'load_hot_reload_configuration'
]