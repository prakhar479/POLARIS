"""
SWIM POLARIS Driver Application

Main driver application that orchestrates the complete SWIM POLARIS adaptation system.
Initializes all components, manages the system lifecycle, and provides monitoring capabilities.
"""

import asyncio
import logging
import signal
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# POLARIS framework imports
from polaris_refactored.src.framework.polaris_framework import PolarisFramework
from polaris_refactored.src.framework.configuration.configuration_builder import ConfigurationBuilder
from polaris_refactored.src.infrastructure.observability import (
    initialize_observability, shutdown_observability, get_logger, get_metrics_collector
)
from polaris_refactored.src.framework.events import TelemetryEvent
from polaris_refactored.src.domain.models import SystemState, HealthStatus

# SWIM-specific imports
from .swim_adaptation_strategies import (
    SwimAdaptationStrategyFactory, SwimAdaptationCoordinator, AdaptationContext, AdaptationTrigger
)
from .swim_metrics_processor import SwimMetricsProcessor, SwimMetricsAggregator


@dataclass
class SystemStatus:
    """Current status of the SWIM POLARIS system."""
    framework_running: bool
    swim_connected: bool
    adaptation_active: bool
    last_adaptation: Optional[datetime]
    total_adaptations: int
    successful_adaptations: int
    failed_adaptations: int
    current_health: HealthStatus
    uptime: float


class SwimPolarisDriver:
    """Main driver for the SWIM POLARIS adaptation system."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        self.framework: Optional[PolarisFramework] = None
        
        # System components
        self.metrics_processor: Optional[SwimMetricsProcessor] = None
        self.metrics_aggregator: Optional[SwimMetricsAggregator] = None
        self.adaptation_strategies: List = []
        self.adaptation_coordinator: Optional[SwimAdaptationCoordinator] = None
        
        # System state
        self.running = False
        self.start_time: Optional[datetime] = None
        self.system_status = SystemStatus(
            framework_running=False,
            swim_connected=False,
            adaptation_active=False,
            last_adaptation=None,
            total_adaptations=0,
            successful_adaptations=0,
            failed_adaptations=0,
            current_health=HealthStatus.UNKNOWN,
            uptime=0.0
        )
        
        # Logging
        self.logger: Optional[logging.Logger] = None
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self._trigger_shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _trigger_shutdown(self):
        """Trigger graceful shutdown."""
        self.shutdown_event.set()
    
    async def initialize(self) -> bool:
        """Initialize the SWIM POLARIS system."""
        try:
            print("Initializing SWIM POLARIS Adaptation System...")
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize observability
            await self._initialize_observability()
            
            # Initialize POLARIS framework
            await self._initialize_framework()
            
            # Initialize SWIM-specific components
            await self._initialize_swim_components()
            
            # Verify SWIM connection
            await self._verify_swim_connection()
            
            self.logger.info("SWIM POLARIS system initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize system: {e}")
            if self.logger:
                self.logger.error(f"System initialization failed: {e}")
            return False
    
    async def _load_configuration(self):
        """Load system configuration."""
        print(f"Loading configuration from {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load additional configuration files if specified
        if "extends" in self.config:
            base_config_path = self.config_path.parent / self.config["extends"]
            if base_config_path.exists():
                with open(base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f)
                # Merge configurations (current config overrides base)
                self.config = self._merge_configs(base_config, self.config)
        
        print("Configuration loaded successfully")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _initialize_observability(self):
        """Initialize observability system."""
        print("Initializing observability...")
        
        observability_config = self.config.get("observability", {})
        
        # Create observability configuration
        from polaris_refactored.src.infrastructure.observability.config_examples import create_observability_config
        obs_config = create_observability_config(
            service_name=observability_config.get("service_name", "swim-polaris-adaptation"),
            environment="development"  # TODO: Make configurable
        )
        
        # Initialize observability
        observability_manager = initialize_observability(obs_config)
        await observability_manager.initialize()
        
        # Get logger
        self.logger = get_logger("swim_driver")
        self.logger.info("Observability system initialized")
    
    async def _initialize_framework(self):
        """Initialize POLARIS framework."""
        self.logger.info("Initializing POLARIS framework...")
        
        # Create configuration builder
        config_builder = ConfigurationBuilder()
        
        # Add configuration sources
        config_builder.add_dict_source(self.config)
        config_builder.add_environment_source("POLARIS_")
        
        # Build configuration
        polaris_config = config_builder.build()
        await polaris_config.load()
        
        # Create and start framework
        self.framework = PolarisFramework(polaris_config)
        await self.framework.start()
        
        self.system_status.framework_running = True
        self.logger.info("POLARIS framework initialized and started")
    
    async def _initialize_swim_components(self):
        """Initialize SWIM-specific components."""
        self.logger.info("Initializing SWIM-specific components...")
        
        # Initialize metrics processor
        metrics_config = self.config.get("metrics_processing", {})
        self.metrics_processor = SwimMetricsProcessor(metrics_config)
        
        # Initialize metrics aggregator
        aggregator_config = self.config.get("metrics_aggregation", {})
        self.metrics_aggregator = SwimMetricsAggregator(aggregator_config)
        
        # Initialize adaptation strategies
        self.adaptation_strategies = SwimAdaptationStrategyFactory.create_strategies_from_config(
            self.config
        )
        
        # Initialize adaptation coordinator
        coordinator_config = self.config.get("control_reasoning", {}).get("adaptive_controller", {})
        self.adaptation_coordinator = SwimAdaptationCoordinator(
            self.adaptation_strategies, coordinator_config
        )
        
        self.logger.info(f"Initialized {len(self.adaptation_strategies)} adaptation strategies")
    
    async def _verify_swim_connection(self):
        """Verify connection to SWIM system."""
        self.logger.info("Verifying SWIM connection...")
        
        try:
            # Get SWIM connector from framework
            # This is a simplified approach - in a full implementation,
            # you would get the connector through the framework's plugin system
            swim_config = self.config.get("managed_systems", [{}])[0].get("config", {})
            
            from polaris_refactored.plugins.swim.connector import SwimTCPConnector
            swim_connector = SwimTCPConnector(swim_config)
            
            # Test connection
            connected = await swim_connector.connect()
            if connected:
                self.system_status.swim_connected = True
                self.logger.info("SWIM connection verified")
            else:
                raise ConnectionError("Failed to connect to SWIM")
                
        except Exception as e:
            self.logger.error(f"SWIM connection verification failed: {e}")
            raise
    
    async def run(self) -> int:
        """Run the SWIM POLARIS system."""
        try:
            # Initialize system
            if not await self.initialize():
                return 1
            
            self.running = True
            self.start_time = datetime.now(timezone.utc)
            self.system_status.adaptation_active = True
            
            self.logger.info("SWIM POLARIS system started successfully")
            print("SWIM POLARIS Adaptation System is running...")
            print("Press Ctrl+C to stop")
            
            # Start main processing loop
            await self._main_loop()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            return 0
        except Exception as e:
            self.logger.error(f"System error: {e}")
            return 1
        finally:
            await self.shutdown()
    
    async def _main_loop(self):
        """Main processing loop."""
        self.logger.info("Starting main processing loop")
        
        # Create tasks for different processing components
        tasks = [
            asyncio.create_task(self._telemetry_processing_loop()),
            asyncio.create_task(self._adaptation_loop()),
            asyncio.create_task(self._status_monitoring_loop()),
            asyncio.create_task(self._wait_for_shutdown())
        ]
        
        try:
            # Wait for any task to complete (usually shutdown)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
            raise
    
    async def _telemetry_processing_loop(self):
        """Process telemetry data from SWIM."""
        self.logger.info("Starting telemetry processing loop")
        
        # Get collection interval from config
        collection_interval = (
            self.config.get("managed_systems", [{}])[0]
            .get("config", {})
            .get("implementation", {})
            .get("collection_interval", 10.0)
        )
        
        while self.running:
            try:
                # Collect telemetry from SWIM
                await self._collect_and_process_telemetry()
                
                # Wait for next collection
                await asyncio.sleep(collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Telemetry processing error: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _collect_and_process_telemetry(self):
        """Collect and process telemetry from SWIM."""
        try:
            # Get SWIM connector (simplified - would normally come from framework)
            swim_config = self.config.get("managed_systems", [{}])[0].get("config", {})
            from polaris_refactored.plugins.swim.connector import SwimTCPConnector
            swim_connector = SwimTCPConnector(swim_config)
            
            # Collect system state
            system_state = await swim_connector.get_system_state()
            
            # Create telemetry event
            telemetry_event = TelemetryEvent(system_state)
            
            # Process metrics
            processed_metrics = await self.metrics_processor.process_telemetry_event(telemetry_event)
            
            # Add to aggregator
            self.metrics_aggregator.add_metrics(processed_metrics.swim_metrics)
            
            # Update system status
            self.system_status.current_health = processed_metrics.health_status
            
            self.logger.debug(f"Processed telemetry: {len(processed_metrics.raw_metrics)} raw metrics, "
                            f"{len(processed_metrics.derived_metrics)} derived metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to collect telemetry: {e}")
            self.system_status.current_health = HealthStatus.UNHEALTHY
    
    async def _adaptation_loop(self):
        """Main adaptation decision and execution loop."""
        self.logger.info("Starting adaptation loop")
        
        # Get adaptation interval from config
        adaptation_interval = (
            self.config.get("control_reasoning", {})
            .get("adaptive_controller", {})
            .get("mape_k_interval", 30.0)
        )
        
        while self.running:
            try:
                await self._execute_adaptation_cycle()
                await asyncio.sleep(adaptation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Adaptation loop error: {e}")
                await asyncio.sleep(10.0)  # Longer pause on error
    
    async def _execute_adaptation_cycle(self):
        """Execute one complete adaptation cycle (MAPE-K)."""
        try:
            # Get recent metrics for context
            recent_metrics = self.metrics_processor.get_recent_metrics(10)
            if not recent_metrics:
                return
            
            current_metrics = recent_metrics[-1]
            
            # Create adaptation context
            context = AdaptationContext(
                current_metrics=current_metrics,
                historical_metrics=recent_metrics[:-1],
                system_state=None,  # Would be populated from framework
                trigger=AdaptationTrigger.THRESHOLD_VIOLATION,
                confidence=0.8,
                constraints=self.config.get("adaptation", {}).get("constraints", {})
            )
            
            # Check if adaptation is needed
            should_adapt, confidence, strategies = await self.adaptation_coordinator.should_adapt(context)
            
            if should_adapt:
                self.logger.info(f"Adaptation needed (confidence: {confidence:.2f}, strategies: {strategies})")
                
                # Plan adaptations
                actions = await self.adaptation_coordinator.plan_adaptations(context)
                
                if actions:
                    # Execute adaptations
                    await self._execute_adaptations(actions)
                    
                    # Update statistics
                    self.system_status.total_adaptations += len(actions)
                    self.system_status.last_adaptation = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Adaptation cycle error: {e}")
    
    async def _execute_adaptations(self, actions):
        """Execute adaptation actions."""
        self.logger.info(f"Executing {len(actions)} adaptation actions")
        
        try:
            # Get SWIM connector
            swim_config = self.config.get("managed_systems", [{}])[0].get("config", {})
            from polaris_refactored.plugins.swim.connector import SwimTCPConnector
            swim_connector = SwimTCPConnector(swim_config)
            
            successful = 0
            failed = 0
            
            for action in actions:
                try:
                    result = await swim_connector.execute_action(action)
                    if result.status.value == "success":
                        successful += 1
                        self.logger.info(f"Action {action.action_type} executed successfully")
                    else:
                        failed += 1
                        self.logger.warning(f"Action {action.action_type} failed: {result.result_data}")
                
                except Exception as e:
                    failed += 1
                    self.logger.error(f"Action {action.action_type} execution error: {e}")
            
            # Update statistics
            self.system_status.successful_adaptations += successful
            self.system_status.failed_adaptations += failed
            
            self.logger.info(f"Adaptation execution completed: {successful} successful, {failed} failed")
            
        except Exception as e:
            self.logger.error(f"Failed to execute adaptations: {e}")
    
    async def _status_monitoring_loop(self):
        """Monitor and report system status."""
        while self.running:
            try:
                await self._update_system_status()
                await self._log_system_status()
                await asyncio.sleep(60.0)  # Status update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Status monitoring error: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_system_status(self):
        """Update system status information."""
        if self.start_time:
            self.system_status.uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    async def _log_system_status(self):
        """Log current system status."""
        status = self.system_status
        
        self.logger.info(
            "System Status",
            extra={
                "framework_running": status.framework_running,
                "swim_connected": status.swim_connected,
                "adaptation_active": status.adaptation_active,
                "total_adaptations": status.total_adaptations,
                "successful_adaptations": status.successful_adaptations,
                "failed_adaptations": status.failed_adaptations,
                "current_health": status.current_health.value,
                "uptime_seconds": status.uptime
            }
        )
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self.shutdown_event.wait()
        self.logger.info("Shutdown signal received")
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        if not self.running:
            return
        
        self.logger.info("Initiating system shutdown...")
        self.running = False
        
        try:
            # Stop framework
            if self.framework:
                await self.framework.stop()
                self.system_status.framework_running = False
            
            # Shutdown observability
            await shutdown_observability()
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        return self.system_status


async def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python swim_driver.py <config_file>")
        return 1
    
    config_file = sys.argv[1]
    driver = SwimPolarisDriver(config_file)
    
    return await driver.run()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)