"""
Connector Factory Module

Factory for creating managed system connectors with configuration support.
"""

import logging
from typing import Dict, List, Optional, Any

from .plugin_registry import PolarisPluginRegistry
from ...domain.interfaces import ManagedSystemConnector
from ...infrastructure.di import Injectable
from ...infrastructure.exceptions import ConnectorError

logger = logging.getLogger(__name__)


class ManagedSystemConnectorFactory(Injectable):
    """
    Factory for creating managed system connectors with configuration support.
    
    Provides a clean interface for creating connectors with proper configuration
    and error handling.
    """
    
    def __init__(self, plugin_registry: PolarisPluginRegistry):
        self.plugin_registry = plugin_registry
        logger.info("ManagedSystemConnectorFactory initialized")
    
    def create_connector(self, system_id: str, system_config: Optional[Dict[str, Any]] = None) -> Optional[ManagedSystemConnector]:
        """Create a connector for the specified system with configuration."""
        try:
            # Load the connector from registry
            connector = self.plugin_registry.load_managed_system_connector(system_id)
            
            if connector and system_config:
                # Configure the connector if configuration is provided
                if hasattr(connector, 'configure'):
                    connector.configure(system_config)
                elif hasattr(connector, '__init__') and system_config:
                    # If connector expects config in constructor, we'd need to recreate it
                    # For now, log a warning
                    logger.warning(
                        f"Connector {system_id} loaded but configuration cannot be applied post-creation"
                    )
            
            return connector
            
        except Exception as e:
            logger.error(f"Failed to create connector {system_id}: {e}")
            raise ConnectorError(
                message=f"Failed to create connector: {system_id}",
                context={"system_id": system_id},
                cause=e
            )
    
    def get_available_connectors(self) -> List[str]:
        """Get list of available connector system IDs."""
        descriptors = self.plugin_registry.get_plugin_descriptors()
        return [plugin_id for plugin_id, desc in descriptors.items() if desc.is_valid]
    
    def get_connector_info(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connector."""
        descriptors = self.plugin_registry.get_plugin_descriptors()
        plugin = descriptors.get(system_id)
        
        if not plugin:
            return None
        
        return {
            "plugin_id": plugin.plugin_id,
            "version": plugin.version,
            "is_valid": plugin.is_valid,
            "is_loaded": self.plugin_registry.is_plugin_loaded(system_id),
            "metadata": plugin.metadata,
            "validation_errors": plugin.validation_errors,
            "path": str(plugin.path)
        }