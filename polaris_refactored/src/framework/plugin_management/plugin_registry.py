"""
Plugin Registry Module

Main registry for managing plugin lifecycle, hot-reloading, and connector management.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .plugin_descriptor import PluginDescriptor
from .plugin_discovery import PluginDiscovery
from ...domain.interfaces import ManagedSystemConnector
from ...infrastructure.di import Injectable
from ...infrastructure.exceptions import ConnectorError

logger = logging.getLogger(__name__)


class PolarisPluginRegistry(Injectable):
    """
    Registry for managing POLARIS plugins with automatic discovery and hot-reloading.
    
    Features:
    - Automatic plugin discovery in configured search paths
    - Plugin validation and error reporting
    - Hot-reloading capabilities without system restart
    - Thread-safe plugin management
    - Comprehensive logging and error handling
    """
    
    def __init__(self):
        self._connectors: Dict[str, ManagedSystemConnector] = {}
        self._plugin_descriptors: Dict[str, PluginDescriptor] = {}
        self._search_paths: List[Path] = []
        self._loaded_modules: Dict[str, Any] = {}
        self._hot_reload_enabled = False
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._stop_hot_reload = threading.Event()
        self._registry_lock = threading.RLock()
        self._discovery = PluginDiscovery()
        
        logger.info("PolarisPluginRegistry initialized")
    
    async def initialize(self, search_paths: Optional[List[Path]] = None, enable_hot_reload: bool = False) -> None:
        """Initialize the plugin registry with discovery and optional hot-reloading."""
        try:
            if search_paths:
                self._search_paths = [Path(p) for p in search_paths]
            
            self._hot_reload_enabled = enable_hot_reload
            
            # Discover and validate plugins
            await self._discover_all_plugins()
            
            # Start hot-reload monitoring if enabled
            if self._hot_reload_enabled:
                self._start_hot_reload_monitoring()
            
            logger.info(
                "Plugin registry initialized",
                extra={
                    "search_paths": [str(p) for p in self._search_paths],
                    "discovered_plugins": len(self._plugin_descriptors),
                    "hot_reload_enabled": self._hot_reload_enabled
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin registry: {e}")
            raise ConnectorError(
                message="Plugin registry initialization failed",
                context={"error": str(e)},
                cause=e
            )
    
    async def shutdown(self) -> None:
        """Shutdown the plugin registry and cleanup resources."""
        try:
            # Stop hot-reload monitoring
            if self._hot_reload_thread:
                self._stop_hot_reload.set()
                self._hot_reload_thread.join(timeout=5.0)
                self._hot_reload_thread = None
            
            # Unload all connectors
            await self.unload_all_connectors()
            
            # Clear registry state
            with self._registry_lock:
                self._plugin_descriptors.clear()
                self._loaded_modules.clear()
            
            logger.info("Plugin registry shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during plugin registry shutdown: {e}")
            raise
    
    async def _discover_all_plugins(self) -> None:
        """Discover all plugins in configured search paths."""
        with self._registry_lock:
            self._plugin_descriptors.clear()
            
            discovered = self._discovery.discover_managed_system_plugins(self._search_paths)
            
            for plugin in discovered:
                self._plugin_descriptors[plugin.plugin_id] = plugin
                
                if plugin.is_valid:
                    logger.info(f"Valid plugin discovered: {plugin.plugin_id} v{plugin.version}")
                else:
                    logger.warning(
                        f"Invalid plugin discovered: {plugin.plugin_id}",
                        extra={"validation_errors": plugin.validation_errors}
                    )
    
    def load_managed_system_connector(self, system_id: str) -> Optional[ManagedSystemConnector]:
        """Load a managed system connector by system ID."""
        with self._registry_lock:
            # Check if already loaded
            if system_id in self._connectors:
                return self._connectors[system_id]
            
            # Find plugin descriptor
            plugin = self._plugin_descriptors.get(system_id)
            if not plugin:
                logger.error(f"Plugin not found: {system_id}")
                return None
            
            if not plugin.is_valid:
                logger.error(
                    f"Cannot load invalid plugin: {system_id}",
                    extra={"validation_errors": plugin.validation_errors}
                )
                return None
            
            try:
                # Load the connector
                connector = self._discovery.load_connector_from_plugin(plugin, self._loaded_modules)
                if connector:
                    self._connectors[system_id] = connector
                    logger.info(f"Connector loaded successfully: {system_id}")
                
                return connector
                
            except Exception as e:
                logger.error(f"Failed to load connector {system_id}: {e}")
                raise ConnectorError(
                    message=f"Failed to load connector: {system_id}",
                    context={"system_id": system_id, "plugin_path": str(plugin.path)},
                    cause=e
                )
    
    async def reload_plugin(self, system_id: str) -> None:
        """Reload a plugin without system restart."""
        with self._registry_lock:
            try:
                # Unload existing connector
                if system_id in self._connectors:
                    connector = self._connectors[system_id]
                    if hasattr(connector, 'disconnect'):
                        try:
                            await connector.disconnect()
                        except Exception as e:
                            logger.warning(f"Error disconnecting connector during reload: {e}")
                    
                    del self._connectors[system_id]
                
                # Remove loaded module
                if system_id in self._loaded_modules:
                    del self._loaded_modules[system_id]
                
                # Rediscover the plugin
                plugin = self._plugin_descriptors.get(system_id)
                if plugin:
                    # Update last modified time and revalidate
                    if plugin.path.is_file():
                        plugin.last_modified = plugin.path.stat().st_mtime
                    else:
                        # For directory plugins, check the main module file
                        module_file = plugin.path / f"{plugin.module_name}.py"
                        if module_file.exists():
                            plugin.last_modified = module_file.stat().st_mtime
                    
                    # Note: Don't reload the connector automatically, just clean up
                    logger.info(f"Plugin unloaded for reload: {system_id}")
                else:
                    logger.error(f"Plugin descriptor not found for reload: {system_id}")
                
            except Exception as e:
                logger.error(f"Error reloading plugin {system_id}: {e}")
                raise ConnectorError(
                    message=f"Failed to reload plugin: {system_id}",
                    context={"system_id": system_id},
                    cause=e
                )
    
    async def load_all_connectors(self) -> None:
        """Load all available and valid connectors."""
        with self._registry_lock:
            valid_plugins = [p for p in self._plugin_descriptors.values() if p.is_valid]
            
            for plugin in valid_plugins:
                if plugin.plugin_id not in self._connectors:
                    try:
                        connector = self.load_managed_system_connector(plugin.plugin_id)
                        if connector:
                            logger.info(f"Auto-loaded connector: {plugin.plugin_id}")
                    except Exception as e:
                        logger.error(f"Failed to auto-load connector {plugin.plugin_id}: {e}")
    
    async def unload_all_connectors(self) -> None:
        """Unload all connectors and cleanup resources."""
        with self._registry_lock:
            for system_id, connector in list(self._connectors.items()):
                try:
                    if hasattr(connector, 'disconnect'):
                        await connector.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting connector {system_id}: {e}")
            
            self._connectors.clear()
            self._loaded_modules.clear()
            logger.info("All connectors unloaded")
    
    def get_loaded_connectors(self) -> Dict[str, ManagedSystemConnector]:
        """Get all currently loaded connectors."""
        with self._registry_lock:
            return self._connectors.copy()
    
    def discover_managed_system_plugins(self, search_paths: List[Path]) -> List[PluginDescriptor]:
        """Discover managed system plugins in the given paths (backward compatibility method)."""
        return self._discovery.discover_managed_system_plugins(search_paths)
    
    def get_plugin_descriptors(self) -> Dict[str, PluginDescriptor]:
        """Get all discovered plugin descriptors."""
        with self._registry_lock:
            return self._plugin_descriptors.copy()
    
    def is_plugin_loaded(self, system_id: str) -> bool:
        """Check if a plugin is currently loaded."""
        with self._registry_lock:
            return system_id in self._connectors
    
    def _start_hot_reload_monitoring(self) -> None:
        """Start hot-reload monitoring in a background thread."""
        if self._hot_reload_thread is not None:
            return
        
        self._stop_hot_reload.clear()
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_worker,
            name="PluginHotReload",
            daemon=True
        )
        self._hot_reload_thread.start()
        logger.info("Hot-reload monitoring started")
    
    def _hot_reload_worker(self) -> None:
        """Background worker for hot-reload monitoring."""
        while not self._stop_hot_reload.is_set():
            try:
                # Check for file changes
                reload_needed = []
                
                with self._registry_lock:
                    for plugin_id, plugin in self._plugin_descriptors.items():
                        if plugin_id in self._connectors:  # Only check loaded plugins
                            current_mtime = self._get_plugin_mtime(plugin)
                            if current_mtime > plugin.last_modified:
                                reload_needed.append(plugin_id)
                
                # Reload changed plugins
                for plugin_id in reload_needed:
                    try:
                        logger.info(f"Plugin file changed, reloading: {plugin_id}")
                        asyncio.run(self.reload_plugin(plugin_id))
                    except Exception as e:
                        logger.error(f"Hot-reload failed for {plugin_id}: {e}")
                
                # Wait before next check
                self._stop_hot_reload.wait(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error in hot-reload monitoring: {e}")
                self._stop_hot_reload.wait(5.0)  # Wait longer on error
    
    def _get_plugin_mtime(self, plugin: PluginDescriptor) -> float:
        """Get the modification time of a plugin's files."""
        try:
            if plugin.path.is_file():
                return plugin.path.stat().st_mtime
            else:
                # For directory plugins, check the main module file
                module_file = plugin.path / f"{plugin.module_name}.py"
                if module_file.exists():
                    return module_file.stat().st_mtime
            return 0.0
        except Exception:
            return 0.0