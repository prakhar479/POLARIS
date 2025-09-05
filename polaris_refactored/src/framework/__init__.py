"""
Framework Layer - Core POLARIS framework services

This layer provides the main entry point and orchestration for the POLARIS system,
including configuration management, plugin management, and the main framework class.
"""

from .polaris_framework import PolarisFramework
from .configuration import PolarisConfiguration, ConfigurationBuilder
from .plugin_management import PolarisPluginRegistry, ManagedSystemConnectorFactory
from .events import PolarisEventBus, PolarisEvent, TelemetryEvent, AdaptationEvent, ExecutionResultEvent

__all__ = [
    "PolarisFramework",
    "PolarisConfiguration", 
    "ConfigurationBuilder",
    "PolarisPluginRegistry",
    "ManagedSystemConnectorFactory", 
    "PolarisEventBus",
    "PolarisEvent",
    "TelemetryEvent",
    "AdaptationEvent", 
    "ExecutionResultEvent",
]