"""
POLARIS Framework - Main orchestration class

This is the main entry point for the POLARIS framework, responsible for
initializing and coordinating all layers of the system.
"""

import asyncio
from typing import Dict, List, Optional
import logging

from ..infrastructure.di import DIContainer, Injectable
from ..infrastructure.exceptions import PolarisException, ConfigurationError
from ..infrastructure.message_bus import PolarisMessageBus
from ..infrastructure.data_storage import PolarisDataStore
from .configuration import PolarisConfiguration
from .plugin_management import PolarisPluginRegistry
from .events import PolarisEventBus


class PolarisFramework(Injectable):
    """
    Main POLARIS Framework class that orchestrates all system components.
    
    This class implements the Facade pattern to provide a simple interface
    for starting, stopping, and managing the entire POLARIS system.
    """
    
    def __init__(
        self,
        container: DIContainer,
        configuration: PolarisConfiguration,
        message_bus: PolarisMessageBus,
        data_store: PolarisDataStore,
        plugin_registry: PolarisPluginRegistry,
        event_bus: PolarisEventBus
    ):
        self.container = container
        self.configuration = configuration
        self.message_bus = message_bus
        self.data_store = data_store
        self.plugin_registry = plugin_registry
        self.event_bus = event_bus
        
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._components: List[str] = []
    
    async def start(self) -> None:
        """
        Start the POLARIS framework and all its components.
        
        This method initializes all layers in the correct order:
        1. Infrastructure layer (message bus, data store)
        2. Framework layer (plugin registry, event bus)
        3. Digital Twin layer
        4. Control & Reasoning layer
        5. Adapter layer
        """
        if self._running:
            self.logger.warning("POLARIS framework is already running")
            return
        
        try:
            self.logger.info("Starting POLARIS framework...")
            
            # Start infrastructure components
            await self._start_infrastructure()
            
            # Start framework components
            await self._start_framework_services()
            
            # Start digital twin components
            await self._start_digital_twin()
            
            # Start control and reasoning components
            await self._start_control_reasoning()
            
            # Start adapter components
            await self._start_adapters()
            
            self._running = True
            self.logger.info("POLARIS framework started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start POLARIS framework: {e}")
            await self._cleanup_on_failure()
            raise PolarisException(
                "Failed to start POLARIS framework",
                error_code="FRAMEWORK_START_ERROR",
                cause=e
            )
    
    async def stop(self) -> None:
        """
        Stop the POLARIS framework and all its components.
        
        Components are stopped in reverse order of startup.
        """
        if not self._running:
            self.logger.warning("POLARIS framework is not running")
            return
        
        try:
            self.logger.info("Stopping POLARIS framework...")
            
            # Stop components in reverse order
            await self._stop_adapters()
            await self._stop_control_reasoning()
            await self._stop_digital_twin()
            await self._stop_framework_services()
            await self._stop_infrastructure()
            
            self._running = False
            self._components.clear()
            
            self.logger.info("POLARIS framework stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping POLARIS framework: {e}")
            raise PolarisException(
                "Failed to stop POLARIS framework",
                error_code="FRAMEWORK_STOP_ERROR",
                cause=e
            )
    
    async def restart(self) -> None:
        """Restart the POLARIS framework."""
        await self.stop()
        await self.start()
    
    def is_running(self) -> bool:
        """Check if the framework is currently running."""
        return self._running
    
    def get_status(self) -> Dict[str, any]:
        """Get the current status of the framework and its components."""
        return {
            "running": self._running,
            "components": self._components.copy(),
            "configuration_loaded": self.configuration is not None,
            "message_bus_connected": hasattr(self.message_bus, '_connected') and self.message_bus._connected,
            "data_store_connected": hasattr(self.data_store, '_connected') and self.data_store._connected
        }
    
    async def _start_infrastructure(self) -> None:
        """Start infrastructure layer components."""
        self.logger.debug("Starting infrastructure layer...")
        
        # Start message bus
        await self.message_bus.start()
        self._components.append("message_bus")
        
        # Start data store
        await self.data_store.start()
        self._components.append("data_store")
        
        self.logger.debug("Infrastructure layer started")
    
    async def _start_framework_services(self) -> None:
        """Start framework layer services."""
        self.logger.debug("Starting framework services...")
        
        # Initialize plugin registry
        await self.plugin_registry.initialize()
        self._components.append("plugin_registry")
        
        # Start event bus
        await self.event_bus.start()
        self._components.append("event_bus")
        
        self.logger.debug("Framework services started")
    
    async def _start_digital_twin(self) -> None:
        """Start digital twin layer components."""
        self.logger.debug("Starting digital twin layer...")
        
        # Resolve and start digital twin components from DI container
        try:
            from ..digital_twin import PolarisWorldModel, PolarisKnowledgeBase, PolarisLearningEngine
            
            # These will be resolved from the DI container when needed
            world_model = self.container.resolve(PolarisWorldModel)
            knowledge_base = self.container.resolve(PolarisKnowledgeBase)
            learning_engine = self.container.resolve(PolarisLearningEngine)
            
            # Start components if they have start methods
            if hasattr(world_model, 'start'):
                await world_model.start()
            if hasattr(knowledge_base, 'start'):
                await knowledge_base.start()
            if hasattr(learning_engine, 'start'):
                await learning_engine.start()
            
            self._components.extend(["world_model", "knowledge_base", "learning_engine"])
            
        except Exception as e:
            self.logger.warning(f"Some digital twin components not available yet: {e}")
        
        self.logger.debug("Digital twin layer started")
    
    async def _start_control_reasoning(self) -> None:
        """Start control and reasoning layer components."""
        self.logger.debug("Starting control and reasoning layer...")
        
        try:
            from ..control_reasoning import PolarisAdaptiveController, PolarisReasoningEngine
            
            # Resolve from DI container
            adaptive_controller = self.container.resolve(PolarisAdaptiveController)
            reasoning_engine = self.container.resolve(PolarisReasoningEngine)
            
            # Start components
            if hasattr(adaptive_controller, 'start'):
                await adaptive_controller.start()
            if hasattr(reasoning_engine, 'start'):
                await reasoning_engine.start()
            
            self._components.extend(["adaptive_controller", "reasoning_engine"])
            
        except Exception as e:
            self.logger.warning(f"Some control/reasoning components not available yet: {e}")
        
        self.logger.debug("Control and reasoning layer started")
    
    async def _start_adapters(self) -> None:
        """Start adapter layer components."""
        self.logger.debug("Starting adapter layer...")
        
        # Load and start managed system connectors
        await self.plugin_registry.load_all_connectors()
        self._components.append("adapters")
        
        self.logger.debug("Adapter layer started")
    
    async def _stop_infrastructure(self) -> None:
        """Stop infrastructure layer components."""
        self.logger.debug("Stopping infrastructure layer...")
        
        if "data_store" in self._components:
            await self.data_store.stop()
            self._components.remove("data_store")
        
        if "message_bus" in self._components:
            await self.message_bus.stop()
            self._components.remove("message_bus")
        
        self.logger.debug("Infrastructure layer stopped")
    
    async def _stop_framework_services(self) -> None:
        """Stop framework layer services."""
        self.logger.debug("Stopping framework services...")
        
        if "event_bus" in self._components:
            await self.event_bus.stop()
            self._components.remove("event_bus")
        
        if "plugin_registry" in self._components:
            await self.plugin_registry.shutdown()
            self._components.remove("plugin_registry")
        
        self.logger.debug("Framework services stopped")
    
    async def _stop_digital_twin(self) -> None:
        """Stop digital twin layer components."""
        self.logger.debug("Stopping digital twin layer...")
        
        # Stop digital twin components if they exist
        for component in ["learning_engine", "knowledge_base", "world_model"]:
            if component in self._components:
                self._components.remove(component)
        
        self.logger.debug("Digital twin layer stopped")
    
    async def _stop_control_reasoning(self) -> None:
        """Stop control and reasoning layer components."""
        self.logger.debug("Stopping control and reasoning layer...")
        
        for component in ["reasoning_engine", "adaptive_controller"]:
            if component in self._components:
                self._components.remove(component)
        
        self.logger.debug("Control and reasoning layer stopped")
    
    async def _stop_adapters(self) -> None:
        """Stop adapter layer components."""
        self.logger.debug("Stopping adapter layer...")
        
        if "adapters" in self._components:
            await self.plugin_registry.unload_all_connectors()
            self._components.remove("adapters")
        
        self.logger.debug("Adapter layer stopped")
    
    async def _cleanup_on_failure(self) -> None:
        """Clean up resources when startup fails."""
        self.logger.debug("Cleaning up after startup failure...")
        
        # Try to stop any components that were started
        try:
            await self.stop()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Factory function for creating a configured POLARIS framework
async def create_polaris_framework(config_path: Optional[str] = None) -> PolarisFramework:
    """
    Factory function to create a fully configured POLARIS framework instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        PolarisFramework: Configured framework instance
    """
    from .configuration import ConfigurationBuilder
    
    # Create DI container
    container = DIContainer()
    
    # Load configuration
    config_builder = ConfigurationBuilder()
    if config_path:
        config_builder.add_yaml_source(config_path)
    config_builder.add_environment_source("POLARIS_")
    
    configuration = config_builder.build()
    
    # Register core services in DI container
    container.register_singleton(PolarisConfiguration, configuration)
    
    # Create and register other components
    # Note: Actual implementations will be created in subsequent tasks
    
    # For now, create a basic framework instance
    framework = PolarisFramework(
        container=container,
        configuration=configuration,
        message_bus=None,  # Will be created in infrastructure tasks
        data_store=None,   # Will be created in infrastructure tasks
        plugin_registry=None,  # Will be created in plugin management tasks
        event_bus=None     # Will be created in event system tasks
    )
    
    return framework