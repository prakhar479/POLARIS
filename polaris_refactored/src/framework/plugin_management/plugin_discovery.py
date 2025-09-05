"""
Plugin Discovery Module

Handles automatic discovery and loading of plugins from configured paths.
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import List, Optional
import yaml

from .plugin_descriptor import PluginDescriptor
from .plugin_validator import PluginValidator
from ...domain.interfaces import ManagedSystemConnector

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """Handles plugin discovery and loading with security validation."""
    
    def __init__(self):
        self._validator = PluginValidator()
    
    def discover_managed_system_plugins(self, search_paths: List[Path]) -> List[PluginDescriptor]:
        """Discover managed system plugins in the given paths."""
        discovered_plugins = []
        
        for search_path in search_paths:
            if not search_path.exists():
                logger.warning(f"Plugin search path does not exist: {search_path}")
                continue
            
            logger.info(f"Discovering plugins in: {search_path}")
            
            # Look for plugin directories or files
            for item in search_path.iterdir():
                if item.is_dir():
                    plugin = self._discover_plugin_in_directory(item)
                    if plugin:
                        discovered_plugins.append(plugin)
                elif item.suffix == '.py' and item.name != '__init__.py':
                    plugin = self._discover_plugin_in_file(item)
                    if plugin:
                        discovered_plugins.append(plugin)
        
        logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def _discover_plugin_in_directory(self, plugin_dir: Path) -> Optional[PluginDescriptor]:
        """Discover a plugin in a directory structure with security validation."""
        try:
            # Security check: Validate plugin path
            path_errors = self._validator.validate_plugin_path(plugin_dir)
            if path_errors:
                logger.warning(f"Plugin path security validation failed for {plugin_dir}: {path_errors}")
                return None
            
            # Look for plugin metadata file
            metadata_file = None
            for name in ['plugin.yaml', 'plugin.yml', 'metadata.yaml', 'metadata.yml']:
                candidate = plugin_dir / name
                if candidate.exists():
                    metadata_file = candidate
                    break
            
            if not metadata_file:
                # Try to find a connector.py file and infer metadata
                connector_file = plugin_dir / 'connector.py'
                if connector_file.exists():
                    return self._infer_plugin_from_connector_file(connector_file)
                return None
            
            # Security check: Validate metadata file path
            metadata_path_errors = self._validator.validate_plugin_path(metadata_file)
            if metadata_path_errors:
                logger.warning(f"Metadata file security validation failed for {metadata_file}: {metadata_path_errors}")
                return None
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
            
            if not isinstance(metadata, dict):
                logger.warning(f"Invalid metadata format in {metadata_file}")
                return None
            
            # Validate metadata with security checks
            validation_errors = self._validator.validate_plugin_metadata(metadata)
            
            # Additional security validation for connector file
            connector_file = plugin_dir / f"{metadata.get('module', 'connector')}.py"
            if connector_file.exists():
                import_errors = self._validator.validate_plugin_imports(connector_file)
                validation_errors.extend(import_errors)
            
            # Create plugin descriptor
            plugin = PluginDescriptor(
                plugin_id=metadata.get('name', plugin_dir.name),
                plugin_type='managed_system_connector',
                version=metadata.get('version', '1.0.0'),
                path=plugin_dir,
                metadata=metadata,
                connector_class_name=metadata.get('connector_class', ''),
                module_name=metadata.get('module', 'connector'),
                last_modified=metadata_file.stat().st_mtime,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors
            )
            
            return plugin
            
        except Exception as e:
            logger.error(f"Error discovering plugin in {plugin_dir}: {e}")
            return None
    
    def _discover_plugin_in_file(self, plugin_file: Path) -> Optional[PluginDescriptor]:
        """Discover a plugin in a single Python file."""
        try:
            return self._infer_plugin_from_connector_file(plugin_file)
        except Exception as e:
            logger.error(f"Error discovering plugin in {plugin_file}: {e}")
            return None
    
    def _infer_plugin_from_connector_file(self, connector_file: Path) -> Optional[PluginDescriptor]:
        """Infer plugin metadata from a connector Python file with security validation."""
        try:
            # Security validation first
            path_errors = self._validator.validate_plugin_path(connector_file)
            import_errors = self._validator.validate_plugin_imports(connector_file)
            
            validation_errors = path_errors + import_errors
            if validation_errors:
                logger.warning(f"Security validation failed for {connector_file}: {validation_errors}")
                # Still create descriptor but mark as invalid
                plugin_id = connector_file.parent.name if connector_file.parent.name != connector_file.parent.parent.name else connector_file.stem
                return PluginDescriptor(
                    plugin_id=plugin_id,
                    plugin_type='managed_system_connector',
                    version='1.0.0',
                    path=connector_file.parent,
                    metadata={'name': plugin_id, 'inferred': True, 'security_failed': True},
                    connector_class_name='',
                    module_name=connector_file.stem,
                    last_modified=connector_file.stat().st_mtime,
                    is_valid=False,
                    validation_errors=validation_errors
                )
            
            # Load the module to inspect it
            spec = importlib.util.spec_from_file_location("temp_connector", connector_file)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find ManagedSystemConnector subclasses
            connector_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj != ManagedSystemConnector and 
                    issubclass(obj, ManagedSystemConnector) and 
                    obj.__module__ == module.__name__):
                    connector_classes.append((name, obj))
            
            if not connector_classes:
                logger.warning(f"No ManagedSystemConnector subclass found in {connector_file}")
                return None
            
            # Use the first connector class found
            class_name, connector_class = connector_classes[0]
            
            # Validate the connector class with enhanced security checks
            class_validation_errors = self._validator.validate_connector_class(connector_class)
            validation_errors.extend(class_validation_errors)
            
            # Use parent directory name as plugin_id if it's in a subdirectory, otherwise use file stem
            if connector_file.parent.name != connector_file.parent.parent.name:
                plugin_id = connector_file.parent.name
            else:
                plugin_id = connector_file.stem
            
            # Create inferred metadata
            metadata = {
                'name': plugin_id,
                'version': '1.0.0',
                'connector_class': class_name,
                'module': connector_file.stem,
                'description': f'Auto-discovered connector from {connector_file.name}',
                'inferred': True
            }
            
            plugin = PluginDescriptor(
                plugin_id=plugin_id,
                plugin_type='managed_system_connector',
                version='1.0.0',
                path=connector_file.parent,
                metadata=metadata,
                connector_class_name=class_name,
                module_name=connector_file.stem,
                last_modified=connector_file.stat().st_mtime,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors
            )
            
            return plugin
            
        except Exception as e:
            logger.error(f"Error inferring plugin from {connector_file}: {e}")
            return None
    
    def load_connector_from_plugin(self, plugin: PluginDescriptor, loaded_modules: dict) -> Optional[object]:
        """Load a connector instance from a plugin descriptor with security validation."""
        try:
            # Determine module path and name
            module_name = plugin.module_name  # Set this first for all code paths
            
            if plugin.path.is_file():
                # Single file plugin
                module_path = plugin.path
            else:
                # Directory plugin
                module_path = plugin.path / f"{plugin.module_name}.py"
            
            # Security validation before loading
            path_errors = self._validator.validate_plugin_path(module_path)
            if path_errors:
                raise ValueError(f"Plugin path validation failed: {path_errors}")
            
            import_errors = self._validator.validate_plugin_imports(module_path)
            if import_errors:
                logger.warning(f"Plugin import validation warnings for {plugin.plugin_id}: {import_errors}")
                # Don't fail on import warnings, just log them
            
            if not module_path.exists():
                raise FileNotFoundError(f"Connector module not found: {module_path}")
            
            # Add plugin directory to Python path if needed
            plugin_dir = plugin.path if plugin.path.is_dir() else plugin.path.parent
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    f"polaris_plugin_{plugin.plugin_id}_{module_name}",
                    module_path
                )
                if not spec or not spec.loader:
                    raise ImportError(f"Cannot create module spec for {module_path}")
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Store module reference for hot-reloading
                loaded_modules[plugin.plugin_id] = module
                
                # Get connector class
                connector_class = getattr(module, plugin.connector_class_name, None)
                if not connector_class:
                    raise AttributeError(f"Connector class not found: {plugin.connector_class_name}")
                
                # Validate the class again (in case it changed)
                validation_errors = self._validator.validate_connector_class(connector_class)
                if validation_errors:
                    raise ValueError(f"Connector validation failed: {validation_errors}")
                
                # Create connector instance
                # Note: We'll need system configuration for full initialization
                # For now, create with minimal config
                connector = connector_class()
                
                logger.info(
                    f"Connector instantiated: {plugin.plugin_id}",
                    extra={
                        "class_name": plugin.connector_class_name,
                        "module_path": str(module_path)
                    }
                )
                
                return connector
                
            finally:
                # Remove from path to avoid conflicts
                if str(plugin_dir) in sys.path:
                    sys.path.remove(str(plugin_dir))
            
        except Exception as e:
            logger.error(f"Error loading connector from plugin {plugin.plugin_id}: {e}")
            raise