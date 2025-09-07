"""
Monitor Adapter Implementation

This module implements the comprehensive monitor adapter, which utilizes the
Strategy pattern for metric collection and integrates with the event system for
telemetry publishing. The adapter provides a flexible and extensible system for
collecting metrics from managed systems and publishing them to the event bus.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ...domain.models import SystemState, HealthStatus
from ...framework.events import PolarisEventBus, TelemetryEvent, EventMetadata
from ...framework.plugin_management import PolarisPluginRegistry, ManagedSystemConnectorFactory
from ..base_adapter import (
    PolarisAdapter, AdapterConfiguration, AdapterHealthStatus, 
    AdapterValidationError
)

from .monitor_types import MonitoringTarget, MetricCollectionMode, CollectionResult
from .monitor_strategy import (
    MetricCollectionStrategy, DirectConnectorStrategy, 
    PollingStrategy, BatchCollectionStrategy, RetryingStrategyDecorator
)

logger = logging.getLogger(__name__)


class MonitorAdapter(PolarisAdapter):
    """
    Monitor adapter for collecting telemetry from managed systems using Strategy pattern.
    
    Features:
    - Pluggable metric collection strategies
    - Integration with plugin system for connector management
    - Telemetry event publishing to event bus
    - Configurable monitoring targets
    - Health monitoring and error handling
    - Performance metrics and monitoring
    """
    
    def __init__(
        self, 
        configuration: AdapterConfiguration,
        event_bus: Optional[PolarisEventBus] = None,
        plugin_registry: Optional[PolarisPluginRegistry] = None
    ):
        super().__init__(configuration, event_bus)
        
        self.plugin_registry = plugin_registry
        self.connector_factory: Optional[ManagedSystemConnectorFactory] = None
        
        # Strategy management
        self._collection_strategies: Dict[str, MetricCollectionStrategy] = {}
        self._default_strategy: Optional[MetricCollectionStrategy] = None
        
        # Monitoring configuration
        self._monitoring_targets: Dict[str, MonitoringTarget] = {}
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._collection_mode = MetricCollectionMode.PULL
        
        # Performance tracking
        self._collection_stats = {
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "average_collection_time": 0.0,
            "last_collection_time": None
        }
        
        logger.info(f"MonitorAdapter {self.adapter_id} initialized")
    
    async def _validate_configuration(self) -> None:
        """Validate monitor adapter configuration."""
        config = self.configuration.config
        
        # Validate collection mode
        if "collection_mode" in config:
            mode_str = config["collection_mode"]
            try:
                self._collection_mode = MetricCollectionMode(mode_str)
            except ValueError:
                raise AdapterValidationError(
                    f"Invalid collection_mode: {mode_str}",
                    self.adapter_id,
                    [f"collection_mode must be one of: {[m.value for m in MetricCollectionMode]}"]
                )
        
        # Validate monitoring targets
        if "monitoring_targets" in config:
            targets_config = config["monitoring_targets"]
            if not isinstance(targets_config, list):
                raise AdapterValidationError(
                    "monitoring_targets must be a list",
                    self.adapter_id,
                    ["monitoring_targets configuration must be a list of target configurations"]
                )
            
            validation_errors = []
            for i, target_config in enumerate(targets_config):
                try:
                    target = MonitoringTarget(**target_config)
                    target_errors = target.validate()
                    if target_errors:
                        validation_errors.extend([f"Target {i}: {error}" for error in target_errors])
                    else:
                        self._monitoring_targets[target.system_id] = target
                except Exception as e:
                    validation_errors.append(f"Target {i}: Invalid configuration - {e}")
            
            if validation_errors:
                raise AdapterValidationError(
                    "Monitoring targets validation failed",
                    self.adapter_id,
                    validation_errors
                )
        
        # Validate plugin registry requirement
        if not self.plugin_registry:
            raise AdapterValidationError(
                "Plugin registry is required for MonitorAdapter",
                self.adapter_id,
                ["plugin_registry must be provided to create connectors"]
            )
        
        logger.info(f"MonitorAdapter {self.adapter_id} configuration validated successfully")
    
    async def _initialize_resources(self) -> None:
        """Initialize monitor adapter resources."""
        # Initialize connector factory
        self.connector_factory = ManagedSystemConnectorFactory(self.plugin_registry)
        
        # Initialize default strategies
        direct_strategy = DirectConnectorStrategy()
        self._collection_strategies[direct_strategy.get_strategy_name()] = direct_strategy
        self._default_strategy = direct_strategy
        
        # Add polling strategy
        polling_strategy = PollingStrategy(direct_strategy)
        self._collection_strategies[polling_strategy.get_strategy_name()] = polling_strategy
        
        # Add batch strategy
        batch_params = {}
        try:
            batch_params = self.configuration.config.get("strategies", {}).get(
                f"batch_{direct_strategy.get_strategy_name()}", {}
            )
        except Exception:
            batch_params = {}
        batch_size = batch_params.get("batch_size", 5)
        batch_strategy = BatchCollectionStrategy(direct_strategy, batch_size=batch_size)
        self._collection_strategies[batch_strategy.get_strategy_name()] = batch_strategy
        
        # Optionally add retrying decorator strategies if enabled via config
        retry_cfg = self.configuration.config.get("strategies", {}).get("retrying", {}) if isinstance(self.configuration.config.get("strategies", {}), dict) else {}
        if retry_cfg.get("enabled", False):
            retry_params = {
                "max_retries": retry_cfg.get("max_retries", 3),
                "backoff_base": retry_cfg.get("backoff_base", 0.5),
                "backoff_factor": retry_cfg.get("backoff_factor", 2.0),
                "max_backoff": retry_cfg.get("max_backoff", 10.0),
                "jitter": retry_cfg.get("jitter", 0.1),
            }
            for base in [direct_strategy, polling_strategy, batch_strategy]:
                wrapped = RetryingStrategyDecorator(base, **retry_params)
                self._collection_strategies[wrapped.get_strategy_name()] = wrapped
        
        # Set default strategy if provided
        default_name = self.configuration.config.get("default_strategy")
        if isinstance(default_name, str) and default_name in self._collection_strategies:
            self._default_strategy = self._collection_strategies[default_name]
        
        # Post-initialization: warn for unknown preferred strategy names in targets
        try:
            for system_id, target in self._monitoring_targets.items():
                preferred = None
                if isinstance(target.config, dict):
                    preferred = target.config.get("collection_strategy")
                if preferred and preferred not in self._collection_strategies:
                    logger.warning(
                        f"Unknown collection_strategy '{preferred}' for system '{system_id}'. "
                        f"Available strategies: {list(self._collection_strategies.keys())}. "
                        f"Falling back to default strategy."
                    )
        except Exception as e:
            logger.error(f"Error during post-initialization strategy validation: {e}")

        logger.info(f"MonitorAdapter {self.adapter_id} resources initialized")
    
    async def _start_processing(self) -> None:
        """Start metric collection processing."""
        # Start PUSH subscriptions if configured
        if self._collection_mode in [MetricCollectionMode.PUSH, MetricCollectionMode.HYBRID]:
            await self._start_push_subscriptions()
        
        if self._collection_mode in [MetricCollectionMode.PULL, MetricCollectionMode.HYBRID]:
            # Start collection tasks for each monitoring target
            for system_id, target in self._monitoring_targets.items():
                if target.enabled:
                    task = asyncio.create_task(
                        self._collection_loop(target),
                        name=f"collection_{system_id}"
                    )
                    self._collection_tasks[system_id] = task
                    logger.info(f"Started collection task for {system_id}")
        
        logger.info(f"MonitorAdapter {self.adapter_id} processing started")
    
    async def _stop_processing(self) -> None:
        """Stop metric collection processing."""
        # Cancel all collection tasks
        for system_id, task in self._collection_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped collection task for {system_id}")
        
        self._collection_tasks.clear()
        
        # Stop PUSH subscriptions if any
        await self._stop_push_subscriptions()
        logger.info(f"MonitorAdapter {self.adapter_id} processing stopped")
    
    async def _cleanup_resources(self) -> None:
        """Clean up monitor adapter resources."""
        # Clear strategies
        self._collection_strategies.clear()
        self._default_strategy = None
        
        # Clear monitoring targets
        self._monitoring_targets.clear()
        
        # Reset connector factory
        self.connector_factory = None
        
        logger.info(f"MonitorAdapter {self.adapter_id} resources cleaned up")
    
    async def _collection_loop(self, target: MonitoringTarget) -> None:
        """Main collection loop for a monitoring target."""
        logger.info(f"Starting collection loop for {target.system_id}")
        
        while self.is_running():
            try:
                # Collect metrics
                result = await self._collect_target_metrics(target)
                
                # Update statistics
                self._update_collection_stats(result)
                
                # Publish telemetry event if successful
                if result.success and self.event_bus:
                    await self._publish_telemetry_event(result)
                
                # Update adapter metrics
                if result.success:
                    self.update_metrics(processed=1)
                else:
                    self.update_metrics(failed=1)
                
                # Compute next interval adaptively (transient per loop)
                base_interval = target.collection_interval
                tc = target.config if isinstance(target.config, dict) else {}
                min_interval = float(tc.get("min_interval", max(1.0, base_interval / 2)))
                max_interval = float(tc.get("max_interval", base_interval * 4))
                success_adjustment = float(tc.get("success_adjustment", 1.0))
                failure_backoff = float(tc.get("failure_backoff", 2.0))
                if result.success:
                    next_interval = max(min_interval, base_interval * success_adjustment)
                else:
                    next_interval = min(max_interval, base_interval * failure_backoff)
                await asyncio.sleep(next_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop for {target.system_id}: {e}")
                self.update_metrics(failed=1)
                
                # Wait before retrying
                await asyncio.sleep(min(target.collection_interval, 30.0))        
        logger.info(f"Collection loop stopped for {target.system_id}")
    
    async def _collect_target_metrics(self, target: MonitoringTarget) -> CollectionResult:
        """Collect metrics for a specific target."""
        # Select appropriate strategy
        strategy = self._select_strategy(target)
        
        if not strategy:
            return CollectionResult(
                system_id=target.system_id,
                metrics={},
                timestamp=datetime.utcnow(),
                success=False,
                error="No suitable collection strategy found",
                strategy_name=None
            )
        
        # Collect metrics using strategy
        result = await strategy.collect_metrics(target, self.connector_factory)
        if result and not result.strategy_name:
            result.strategy_name = strategy.get_strategy_name()
        return result
    
    def _select_strategy(self, target: MonitoringTarget) -> Optional[MetricCollectionStrategy]:
        """Select the appropriate collection strategy for a target."""
        # Check if target specifies a preferred strategy
        preferred_strategy = target.config.get("collection_strategy")
        if preferred_strategy and preferred_strategy in self._collection_strategies:
            strategy = self._collection_strategies[preferred_strategy]
            if strategy.supports_target(target):
                return strategy
        
        # Find first strategy that supports the target
        for strategy in self._collection_strategies.values():
            if strategy.supports_target(target):
                return strategy
        
        # Fall back to default strategy
        return self._default_strategy
    
    async def _publish_telemetry_event(self, result: CollectionResult) -> None:
        """Publish telemetry event to event bus."""
        try:
            # Create system state from collection result
            system_state = SystemState(
                system_id=result.system_id,
                health_status=HealthStatus.HEALTHY if result.success else HealthStatus.UNHEALTHY,
                metrics=result.metrics,
                timestamp=result.timestamp
            )
            
            # Create telemetry event
            event = TelemetryEvent(
                system_state=system_state,
                metadata=EventMetadata(
                    source=f"monitor_adapter:{self.adapter_id}",
                    tags={
                        "system_id": result.system_id,
                        "collection_strategy": (result.strategy_name or "unknown"),
                        "collection_success": "true" if result.success else "false",
                        "collection_duration_seconds": str(result.collection_duration.total_seconds()) if result.collection_duration else "0"
                    }
                )
            )
            
            # Publish event
            await self.event_bus.publish_telemetry(event)
            
            logger.debug(f"Published telemetry event for {result.system_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish telemetry event for {result.system_id}: {e}")

    async def _start_push_subscriptions(self) -> None:
        """Start push-based telemetry subscriptions if connectors support it."""
        if not hasattr(self, "_push_subscriptions"):
            self._push_subscriptions: Dict[str, Any] = {}
        for system_id, target in self._monitoring_targets.items():
            if not target.enabled:
                continue
            try:
                connector = self.connector_factory.create_connector(target.connector_type, target.config)
                if hasattr(connector, "subscribe_telemetry") and callable(getattr(connector, "subscribe_telemetry")):
                    async def handler(payload):
                        try:
                            # Attempt to interpret payload
                            if isinstance(payload, SystemState):
                                state = payload
                            elif isinstance(payload, dict):
                                metrics = payload.get("metrics", {})
                                ts = payload.get("timestamp", datetime.utcnow())
                                state = SystemState(
                                    system_id=system_id,
                                    health_status=HealthStatus.HEALTHY,
                                    metrics=metrics,
                                    timestamp=ts,
                                )
                            else:
                                # Fallback: publish empty metrics with warning
                                state = SystemState(
                                    system_id=system_id,
                                    health_status=HealthStatus.HEALTHY,
                                    metrics={},
                                    timestamp=datetime.utcnow(),
                                )
                            event = TelemetryEvent(
                                system_state=state,
                                metadata=EventMetadata(
                                    source=f"monitor_adapter:{self.adapter_id}",
                                    tags={
                                        "system_id": system_id,
                                        "collection_strategy": "push"
                                    }
                                )
                            )
                            await self.event_bus.publish_telemetry(event)
                        except Exception as ex:
                            logger.error(f"Push telemetry handler error for {system_id}: {ex}")
                    token = connector.subscribe_telemetry(handler)
                    self._push_subscriptions[system_id] = {
                        "connector": connector,
                        "token": token,
                    }
                    logger.info(f"Subscribed to push telemetry for {system_id}")
                else:
                    logger.warning(f"Connector for {system_id} does not support push telemetry subscription")
            except Exception as e:
                logger.error(f"Failed to start push subscription for {system_id}: {e}")

    async def _stop_push_subscriptions(self) -> None:
        """Stop push-based telemetry subscriptions if previously started."""
        subs = getattr(self, "_push_subscriptions", {})
        for system_id, entry in list(subs.items()):
            connector = entry.get("connector")
            token = entry.get("token")
            try:
                if connector and hasattr(connector, "unsubscribe") and callable(getattr(connector, "unsubscribe")) and token is not None:
                    connector.unsubscribe(token)
                    logger.info(f"Unsubscribed push telemetry for {system_id}")
            except Exception as e:
                logger.error(f"Failed to stop push subscription for {system_id}: {e}")
            finally:
                subs.pop(system_id, None)
    
    def _update_collection_stats(self, result: CollectionResult) -> None:
        """Update collection statistics."""
        self._collection_stats["total_collections"] += 1
        
        if result.success:
            self._collection_stats["successful_collections"] += 1
        else:
            self._collection_stats["failed_collections"] += 1
        
        if result.collection_duration:
            # Update average collection time
            total_time = (self._collection_stats["average_collection_time"] * 
                         (self._collection_stats["total_collections"] - 1) + 
                         result.collection_duration.total_seconds())
            self._collection_stats["average_collection_time"] = total_time / self._collection_stats["total_collections"]
        
        self._collection_stats["last_collection_time"] = result.timestamp
    
    async def _check_health(self) -> AdapterHealthStatus:
        """Perform monitor adapter specific health check."""
        try:
            # Check if we have active collection tasks
            if not self._collection_tasks:
                return AdapterHealthStatus.DEGRADED
            
            # Check collection success rate
            total = self._collection_stats["total_collections"]
            if total > 0:
                success_rate = self._collection_stats["successful_collections"] / total
                if success_rate < 0.5:  # Less than 50% success
                    return AdapterHealthStatus.UNHEALTHY
                elif success_rate < 0.8:  # Less than 80% success
                    return AdapterHealthStatus.DEGRADED
            
            # Check if collections are happening recently
            last_collection = self._collection_stats["last_collection_time"]
            if last_collection:
                time_since_last = datetime.utcnow() - last_collection
                if time_since_last > timedelta(minutes=5):  # No collections in 5 minutes
                    return AdapterHealthStatus.DEGRADED
            
            return AdapterHealthStatus.HEALTHY
            
        except Exception as e:
            logger.error(f"Health check failed for MonitorAdapter {self.adapter_id}: {e}")
            return AdapterHealthStatus.UNHEALTHY
    
    def add_monitoring_target(self, target: MonitoringTarget) -> None:
        """Add a new monitoring target."""
        validation_errors = target.validate()
        if validation_errors:
            raise ValueError(f"Invalid monitoring target: {validation_errors}")
        
        self._monitoring_targets[target.system_id] = target
        
        # Start collection task if adapter is running
        if self.is_running() and target.enabled:
            task = asyncio.create_task(
                self._collection_loop(target),
                name=f"collection_{target.system_id}"
            )
            self._collection_tasks[target.system_id] = task
            logger.info(f"Added monitoring target and started collection for {target.system_id}")
    
    def remove_monitoring_target(self, system_id: str) -> bool:
        """Remove a monitoring target."""
        if system_id not in self._monitoring_targets:
            return False
        
        # Stop collection task if running
        if system_id in self._collection_tasks:
            task = self._collection_tasks[system_id]
            task.cancel()
            del self._collection_tasks[system_id]
        
        # Remove target
        del self._monitoring_targets[system_id]
        logger.info(f"Removed monitoring target {system_id}")
        return True
    
    def get_monitoring_targets(self) -> Dict[str, MonitoringTarget]:
        """Get all monitoring targets."""
        return self._monitoring_targets.copy()
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = self._collection_stats.copy()
        stats.update({
            "active_targets": len(self._monitoring_targets),
            "active_collection_tasks": len(self._collection_tasks),
            "available_strategies": list(self._collection_strategies.keys()),
            "collection_mode": self._collection_mode.value
        })
        return stats
    
    def add_collection_strategy(self, strategy: MetricCollectionStrategy) -> None:
        """Add a new collection strategy."""
        self._collection_strategies[strategy.get_strategy_name()] = strategy
        logger.info(f"Added collection strategy: {strategy.get_strategy_name()}")
    
    def set_default_strategy(self, strategy_name: str) -> bool:
        """Set the default collection strategy."""
        if strategy_name not in self._collection_strategies:
            return False
        
        self._default_strategy = self._collection_strategies[strategy_name]
        logger.info(f"Set default collection strategy to: {strategy_name}")
        return True