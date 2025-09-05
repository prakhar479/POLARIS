"""
Core configuration management class.
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List, Callable

from ...infrastructure.di import Injectable
from .models import FrameworkConfiguration, ManagedSystemConfiguration
from .sources import ConfigurationSource, YAMLConfigurationSource
from .validation import ConfigurationValidator

logger = logging.getLogger(__name__)


class PolarisConfiguration(Injectable):
    """
    Main configuration class for POLARIS with hierarchical configuration management and hot-reloading.
    
    Supports multiple configuration sources with priority-based merging, validation, and hot-reloading.
    """
    
    def __init__(self, sources: Optional[List[ConfigurationSource]] = None, enable_hot_reload: bool = False):
        self._sources = sources or []
        self._config_data: Dict[str, Any] = {}
        self._framework_config: Optional[FrameworkConfiguration] = None
        self._managed_systems: Dict[str, ManagedSystemConfiguration] = {}
        self._enable_hot_reload = enable_hot_reload
        self._reload_callbacks: List[Callable[[], None]] = []
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._stop_hot_reload = threading.Event()
        self._config_lock = threading.RLock()
        
        if self._sources:
            self._load_configuration()
        
        if self._enable_hot_reload:
            self._start_hot_reload_monitoring()
    
    def add_source(self, source: ConfigurationSource) -> None:
        """Add a configuration source."""
        with self._config_lock:
            self._sources.append(source)
            self._sources.sort(key=lambda s: s.get_priority())
    
    def _load_configuration(self) -> None:
        """Load and merge configuration from all sources."""
        merged_config = {}
        
        # Load from sources in priority order (lowest to highest)
        for source in sorted(self._sources, key=lambda s: s.get_priority()):
            try:
                source_config = source.load()
                merged_config = self._deep_merge(merged_config, source_config)
            except Exception as e:
                logger.error(f"Failed to load configuration from source: {type(source).__name__}: {e}")
                raise
        
        # Validate the merged configuration
        try:
            ConfigurationValidator.validate_configuration(merged_config)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        with self._config_lock:
            self._config_data = merged_config
            self._parse_configurations()
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _parse_configurations(self) -> None:
        """Parse and validate configuration objects."""
        try:
            # Parse framework configuration
            framework_data = self._config_data.get('framework', {})
            self._framework_config = FrameworkConfiguration(**framework_data)
            
            # Parse managed system configurations
            managed_systems_data = self._config_data.get('managed_systems', {})
            self._managed_systems = {}
            
            for system_id, system_config in managed_systems_data.items():
                if isinstance(system_config, dict):
                    system_config['system_id'] = system_id
                    self._managed_systems[system_id] = ManagedSystemConfiguration(**system_config)
                    
        except Exception as e:
            logger.error(f"Failed to parse configurations: {e}")
            raise
    
    def get_framework_config(self) -> FrameworkConfiguration:
        """Get framework configuration."""
        with self._config_lock:
            if self._framework_config is None:
                # Return default configuration if none loaded
                return FrameworkConfiguration()
            return self._framework_config
    
    def get_managed_system_config(self, system_id: str) -> Optional[ManagedSystemConfiguration]:
        """Get configuration for a specific managed system."""
        with self._config_lock:
            return self._managed_systems.get(system_id)
    
    def get_all_managed_systems(self) -> Dict[str, ManagedSystemConfiguration]:
        """Get all managed system configurations."""
        with self._config_lock:
            return self._managed_systems.copy()
    
    def reload_configuration(self) -> None:
        """Reload configuration from all sources."""
        try:
            self._load_configuration()
            
            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration data."""
        with self._config_lock:
            return self._config_data.copy()
    
    def add_reload_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called when configuration is reloaded."""
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[], None]) -> None:
        """Remove a reload callback."""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def _start_hot_reload_monitoring(self) -> None:
        """Start hot-reload monitoring in a background thread."""
        if self._hot_reload_thread is not None:
            return
        
        self._stop_hot_reload.clear()
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_worker,
            name="ConfigHotReload",
            daemon=True
        )
        self._hot_reload_thread.start()
    
    def _hot_reload_worker(self) -> None:
        """Background worker for hot-reload monitoring."""
        while not self._stop_hot_reload.is_set():
            try:
                # Check if any YAML sources have changed
                reload_needed = False
                for source in self._sources:
                    if isinstance(source, YAMLConfigurationSource):
                        if source.has_changed():
                            reload_needed = True
                            break
                
                if reload_needed:
                    logger.info("Configuration file changed, reloading...")
                    self.reload_configuration()
                
                # Wait before next check
                self._stop_hot_reload.wait(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in hot-reload monitoring: {e}")
                self._stop_hot_reload.wait(5.0)  # Wait longer on error
    
    def stop_hot_reload(self) -> None:
        """Stop hot-reload monitoring."""
        if self._hot_reload_thread is not None:
            self._stop_hot_reload.set()
            self._hot_reload_thread.join(timeout=5.0)
            self._hot_reload_thread = None
            logger.info("Hot-reload monitoring stopped")
    
    def is_hot_reload_enabled(self) -> bool:
        """Check if hot-reload is enabled."""
        return self._enable_hot_reload and self._hot_reload_thread is not None
    
    def __del__(self):
        """Cleanup when the configuration object is destroyed."""
        self.stop_hot_reload()