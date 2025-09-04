"""
Core base classes for POLARIS adapters and connectors.

This module provides the clean, single inheritance system for all POLARIS
adapters and external system connectors.
"""

import abc
import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from polaris.common.config import ConfigurationManager
from polaris.common.nats_client import NATSClient


class ManagedSystemConnector(abc.ABC):
    """Abstract base class for all managed system connectors.
    
    This interface defines the contract that all managed system
    connectors must implement to integrate with POLARIS.
    """
    
    def __init__(self, system_config: Dict[str, Any], logger: logging.Logger):
        """Initialize the connector.
        
        Args:
            system_config: Complete configuration for the managed system
            logger: Logger instance for structured logging
        """
        self.config = system_config
        self.logger = logger
        self.connection_config = system_config.get("connection", {})
        self.implementation_config = system_config.get("implementation", {})
        
    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the managed system."""
        pass
    
    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the managed system."""
        pass
    
    @abc.abstractmethod
    async def execute_command(
        self,
        command_template: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a command on the managed system."""
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the managed system is healthy and responsive."""
        pass
    
    async def validate_connection(self) -> bool:
        """Validate that the connection is still active."""
        try:
            return await self.health_check()
        except Exception as e:
            self.logger.warning(f"Connection validation failed: {e}")
            return False
    
    def get_timeout(self) -> float:
        """Get the configured timeout for operations."""
        return self.implementation_config.get("timeout", 30.0)
    
    def get_max_retries(self) -> int:
        """Get the configured maximum retries for operations."""
        return self.implementation_config.get("max_retries", 3)


class BaseComponent(abc.ABC):
    """
    Base class for all POLARIS components.
    
    Provides common functionality including NATS communication,
    configuration management, and lifecycle management.
    """
    
    def __init__(
        self,
        polaris_config_path: str,
        logger: Optional[logging.Logger] = None,
        component_name: Optional[str] = None
    ):
        """Initialize the base component."""
        self.logger = logger or logging.getLogger(
            component_name or self.__class__.__name__
        )
        
        # Initialize configuration manager
        self.config_manager = ConfigurationManager(self.logger)
        
        # Load framework configuration
        self.framework_config = self.config_manager.load_framework_config(
            polaris_config_path
        )
        
        # Initialize NATS client
        nats_config = self.framework_config.get("nats", {})
        self.nats_client = NATSClient(
            nats_url=nats_config.get("url", "nats://localhost:4222"),
            logger=self.logger,
            name=component_name or self.__class__.__name__
        )
        
        # Runtime state
        self.running = False
        self._tasks = []
        
        self.logger.info(f"{self.__class__.__name__} initialized")
    
    async def start(self) -> None:
        """Start the component."""
        self.logger.info(f"Starting {self.__class__.__name__}")
        
        # Connect to NATS
        await self.nats_client.connect()
        
        # Set running flag
        self.running = True
        
        # Start component-specific processing
        await self._start_processing()
        
        self.logger.info(f"{self.__class__.__name__} started")
    
    async def stop(self) -> None:
        """Stop the component gracefully."""
        self.logger.info(f"Stopping {self.__class__.__name__}")
        
        # Clear running flag
        self.running = False
        
        # Stop component-specific processing
        await self._stop_processing()
        
        # Cancel any running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close NATS connection
        await self.nats_client.close()
        
        self.logger.info(f"{self.__class__.__name__} stopped")
    
    @abc.abstractmethod
    async def _start_processing(self) -> None:
        """Start component-specific processing."""
        pass
    
    @abc.abstractmethod
    async def _stop_processing(self) -> None:
        """Stop component-specific processing."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class ExternalAdapter(BaseComponent):
    """
    Base class for adapters that interface with external managed systems.
    
    These adapters require system-specific connectors and plugin configurations.
    Examples: Monitor Adapter, Execution Adapter
    """
    
    def __init__(
        self,
        polaris_config_path: str,
        plugin_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the external adapter."""
        super().__init__(polaris_config_path, logger)
        
        self.plugin_dir = Path(plugin_dir)
        
        # Load schema for validation
        schema_path = Path(__file__).parent.parent.parent / "config" / "managed_system.schema.json"
        if schema_path.exists():
            self.config_manager.load_schema(schema_path)
        
        # Load plugin configuration (required for external adapters)
        self.plugin_config = self.config_manager.load_plugin_config(
            self.plugin_dir,
            validate=True
        )
        
        # Load and initialize the managed system connector
        self.connector = self._load_connector()
        
        self.logger.info(
            f"{self.__class__.__name__} external adapter initialized",
            extra={
                "system_name": self.plugin_config.get("system_name"),
                "connector_class": self.connector.__class__.__name__
            }
        )
    
    def _load_connector(self) -> ManagedSystemConnector:
        """Load and instantiate the managed system connector."""
        # Add plugin directory to Python path
        if str(self.plugin_dir) not in sys.path:
            sys.path.insert(0, str(self.plugin_dir))
        
        # Get connector class path
        connector_path = self.config_manager.get_plugin_connector_class()
        
        # Split module and class name
        if '.' in connector_path:
            module_path, class_name = connector_path.rsplit('.', 1)
        else:
            module_path = "connector"
            class_name = connector_path
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the connector class
            connector_class = getattr(module, class_name, None)
            if connector_class is None:
                raise ValueError(f"Class {class_name} not found in module {module_path}")
            
            # Verify it's a subclass of ManagedSystemConnector
            if not issubclass(connector_class, ManagedSystemConnector):
                raise ValueError(
                    f"{class_name} must be a subclass of ManagedSystemConnector"
                )
            
            # Instantiate the connector
            connector = connector_class(
                system_config=self.plugin_config,
                logger=self.logger
            )
            
            self.logger.info(f"Connector loaded: {class_name}")
            return connector
            
        except Exception as e:
            self.logger.error(f"Failed to load connector: {e}")
            raise
    
    async def start(self) -> None:
        """Start the external adapter."""
        await super().start()
        
        # Connect to managed system
        await self.connector.connect()
        
        self.logger.info(f"{self.__class__.__name__} connected to managed system")
    
    async def stop(self) -> None:
        """Stop the external adapter."""
        # Disconnect from managed system first
        try:
            await self.connector.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting from managed system: {e}")
        
        # Then stop the base component
        await super().stop()


class InternalAdapter(BaseComponent):
    """
    Base class for internal POLARIS framework adapters.
    
    These adapters are part of the framework and don't interface with
    external systems. They can optionally load plugin configurations.
    Examples: Verification Adapter
    """
    
    def __init__(
        self,
        polaris_config_path: str,
        plugin_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        component_name: Optional[str] = None
    ):
        """Initialize the internal adapter."""
        super().__init__(polaris_config_path, logger, component_name)
        
        self.plugin_dir = Path(plugin_dir) if plugin_dir else None
        self.plugin_config = {}
        
        # Load plugin configuration if plugin directory is provided
        if self.plugin_dir:
            self._load_plugin_configuration()
        
        self.logger.info(
            f"{self.__class__.__name__} internal adapter initialized",
            extra={
                "plugin_dir": str(self.plugin_dir) if self.plugin_dir else None,
                "has_plugin_config": bool(self.plugin_config)
            }
        )
    
    def _load_plugin_configuration(self) -> None:
        """Load plugin configuration for internal adapter use."""
        try:
            self.plugin_config = self.config_manager.load_plugin_config(
                self.plugin_dir,
                validate=False  # Internal adapters may not need full validation
            )
            
            self.logger.info(
                "Plugin configuration loaded",
                extra={
                    "system_name": self.plugin_config.get("system_name", "unknown")
                }
            )
            
        except Exception as e:
            self.logger.warning(
                f"Could not load plugin configuration: {e} - using framework defaults"
            )
            self.plugin_config = {}
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        # Try plugin config first, then framework config
        if self.plugin_config and component_name in self.plugin_config:
            return self.plugin_config[component_name]
        
        return self.framework_config.get(component_name, {})