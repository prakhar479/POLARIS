"""
Plugin Management System

Provides plugin discovery, loading, and lifecycle management for POLARIS managed system connectors.
Supports automatic plugin discovery, hot-reloading, and validation.

This module serves as a backward compatibility layer and imports from the modular plugin_management package.
"""

import logging

# Import all components from the modular plugin_management package
from .plugin_management import (
    PluginDescriptor,
    PluginValidator,
    PluginDiscovery,
    PolarisPluginRegistry,
    ManagedSystemConnectorFactory
)

logger = logging.getLogger(__name__)

# Maintain backward compatibility by exposing all classes at module level
__all__ = [
    'PluginDescriptor',
    'PluginValidator',
    'PluginDiscovery', 
    'PolarisPluginRegistry',
    'ManagedSystemConnectorFactory'
]

logger.info("Plugin management system loaded with modular architecture")