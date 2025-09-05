"""
Plugin Management System for POLARIS

This package provides comprehensive plugin discovery, loading, and lifecycle management
for POLARIS managed system connectors with security validation and hot-reloading capabilities.
"""

from .plugin_descriptor import PluginDescriptor
from .plugin_validator import PluginValidator
from .plugin_discovery import PluginDiscovery
from .plugin_registry import PolarisPluginRegistry
from .connector_factory import ManagedSystemConnectorFactory

__all__ = [
    'PluginDescriptor',
    'PluginValidator', 
    'PluginDiscovery',
    'PolarisPluginRegistry',
    'ManagedSystemConnectorFactory'
]