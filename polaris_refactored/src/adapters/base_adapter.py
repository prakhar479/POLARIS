"""
Base Adapter Implementation

Comprehensive base adapter using Template Method pattern with lifecycle management,
validation hooks, error handling, and integration with POLARIS framework components.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..infrastructure.di import Injectable
from ..infrastructure.exceptions import AdaptationError, PolarisException
from ..framework.events import PolarisEventBus

logger = logging.getLogger(__name__)


class AdapterState(Enum):
    """States of adapter lifecycle."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class AdapterHealthStatus(Enum):
    """Health status of adapter."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AdapterMetrics:
    """Metrics collected by adapter."""
    processed_items: int = 0
    failed_items: int = 0
    last_activity: Optional[datetime] = None
    uptime: Optional[timedelta] = None
    error_rate: float = 0.0
    throughput: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.processed_items + self.failed_items
        return (self.processed_items / total) if total > 0 else 0.0


@dataclass
class AdapterConfiguration:
    """Configuration for adapter with validation."""
    adapter_id: str
    adapter_type: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    timeout: float = 30.0  # seconds
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.adapter_id or not isinstance(self.adapter_id, str):
            errors.append("adapter_id must be a non-empty string")
        
        if not self.adapter_type or not isinstance(self.adapter_type, str):
            errors.append("adapter_type must be a non-empty string")
        
        if not isinstance(self.enabled, bool):
            errors.append("enabled must be a boolean")
        
        if not isinstance(self.config, dict):
            errors.append("config must be a dictionary")
        
        if self.health_check_interval <= 0:
            errors.append("health_check_interval must be positive")
        
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        
        if self.retry_delay < 0:
            errors.append("retry_delay must be non-negative")
        
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        
        return errors


class AdapterValidationError(PolarisException):
    """Exception raised when adapter validation fails."""
    
    def __init__(self, message: str, adapter_id: str, validation_errors: List[str]):
        context = {
            "adapter_id": adapter_id,
            "validation_errors": validation_errors
        }
        super().__init__(
            message=message,
            error_code="ADAPTER_VALIDATION_ERROR",
            context=context
        )
        self.adapter_id = adapter_id
        self.validation_errors = validation_errors


class AdapterLifecycleError(AdaptationError):
    """Exception raised during adapter lifecycle operations."""
    
    def __init__(self, message: str, adapter_id: str, state: AdapterState, cause: Exception = None):
        context = {
            "adapter_id": adapter_id,
            "state": state.value
        }
        super().__init__(
            message=message,
            context=context,
            cause=cause
        )
        self.adapter_id = adapter_id
        self.state = state


class PolarisAdapter(Injectable, ABC):
    """
    Base class for all POLARIS adapters using Template Method pattern.
    
    Features:
    - Comprehensive lifecycle management with state tracking
    - Configuration validation with detailed error reporting
    - Health monitoring and metrics collection
    - Error handling with retry mechanisms
    - Integration with event system for monitoring
    - Graceful shutdown with resource cleanup
    - Extensible hook system for customization
    """
    
    def __init__(
        self, 
        configuration: AdapterConfiguration,
        event_bus: Optional[PolarisEventBus] = None
    ):
        self.configuration = configuration
        self.event_bus = event_bus
        
        # State management
        self._state = AdapterState.STOPPED
        self._health_status = AdapterHealthStatus.UNKNOWN
        self._start_time: Optional[datetime] = None
        self._last_health_check: Optional[datetime] = None
        
        # Metrics and monitoring
        self._metrics = AdapterMetrics()
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Error handling
        self._retry_count = 0
        self._last_error: Optional[Exception] = None
        
        # Lifecycle hooks
        self._pre_start_hooks: List[Callable] = []
        self._post_start_hooks: List[Callable] = []
        self._pre_stop_hooks: List[Callable] = []
        self._post_stop_hooks: List[Callable] = []
        
        logger.info(f"Created adapter {self.adapter_id} of type {self.configuration.adapter_type}")
    
    @property
    def adapter_id(self) -> str:
        """Get adapter ID."""
        return self.configuration.adapter_id
    
    @property
    def adapter_type(self) -> str:
        """Get adapter type."""
        return self.configuration.adapter_type
    
    @property
    def state(self) -> AdapterState:
        """Get current adapter state."""
        return self._state
    
    @property
    def health_status(self) -> AdapterHealthStatus:
        """Get current health status."""
        return self._health_status
    
    @property
    def metrics(self) -> AdapterMetrics:
        """Get adapter metrics."""
        if self._start_time:
            self._metrics.uptime = datetime.utcnow() - self._start_time
        return self._metrics
    
    def is_running(self) -> bool:
        """Check if the adapter is running."""
        return self._state == AdapterState.RUNNING
    
    def is_healthy(self) -> bool:
        """Check if the adapter is healthy."""
        return self._health_status in [AdapterHealthStatus.HEALTHY, AdapterHealthStatus.DEGRADED]
    
    async def start(self) -> None:
        """
        Start the adapter using template method pattern.
        
        Template method that orchestrates the startup process:
        1. Validate configuration
        2. Run pre-start hooks
        3. Initialize resources
        4. Start processing
        5. Start health monitoring
        6. Run post-start hooks
        """
        if self._state in [AdapterState.STARTING, AdapterState.RUNNING]:
            logger.warning(f"Adapter {self.adapter_id} is already starting or running")
            return
        
        if not self.configuration.enabled:
            logger.info(f"Adapter {self.adapter_id} is disabled, skipping start")
            return
        
        logger.info(f"Starting adapter {self.adapter_id}")
        self._state = AdapterState.STARTING
        
        try:
            # Step 1: Validate configuration
            await self._validate_configuration_internal()
            
            # Step 2: Run pre-start hooks
            await self._run_hooks(self._pre_start_hooks, "pre-start")
            
            # Step 3: Initialize resources
            await self._initialize_resources_internal()
            
            # Step 4: Start processing
            await self._start_processing_internal()
            
            # Step 5: Start health monitoring
            await self._start_health_monitoring()
            
            # Step 6: Update state and metrics
            self._state = AdapterState.RUNNING
            self._health_status = AdapterHealthStatus.HEALTHY
            self._start_time = datetime.utcnow()
            self._retry_count = 0
            self._last_error = None
            
            # Step 7: Run post-start hooks
            await self._run_hooks(self._post_start_hooks, "post-start")
            
            logger.info(f"Adapter {self.adapter_id} started successfully")
            
            # Publish start event
            await self._publish_lifecycle_event("started")
            
        except Exception as e:
            self._state = AdapterState.ERROR
            self._health_status = AdapterHealthStatus.UNHEALTHY
            self._last_error = e
            
            logger.error(f"Failed to start adapter {self.adapter_id}: {e}")
            
            # Attempt cleanup
            try:
                await self._cleanup_resources_internal()
            except Exception as cleanup_error:
                logger.error(f"Error during startup cleanup for {self.adapter_id}: {cleanup_error}")
            
            # Publish error event
            await self._publish_lifecycle_event("start_failed", {"error": str(e)})
            
            raise AdapterLifecycleError(
                f"Failed to start adapter {self.adapter_id}",
                self.adapter_id,
                self._state,
                e
            )
    
    async def stop(self) -> None:
        """
        Stop the adapter gracefully.
        
        Template method that orchestrates the shutdown process:
        1. Run pre-stop hooks
        2. Stop health monitoring
        3. Stop processing
        4. Cleanup resources
        5. Run post-stop hooks
        """
        if self._state in [AdapterState.STOPPED, AdapterState.STOPPING]:
            logger.warning(f"Adapter {self.adapter_id} is already stopped or stopping")
            return
        
        logger.info(f"Stopping adapter {self.adapter_id}")
        self._state = AdapterState.STOPPING
        
        try:
            # Step 1: Run pre-stop hooks
            await self._run_hooks(self._pre_stop_hooks, "pre-stop")
            
            # Step 2: Stop health monitoring
            await self._stop_health_monitoring()
            
            # Step 3: Stop processing
            await self._stop_processing_internal()
            
            # Step 4: Cleanup resources
            await self._cleanup_resources_internal()
            
            # Step 5: Update state
            self._state = AdapterState.STOPPED
            self._health_status = AdapterHealthStatus.UNKNOWN
            self._start_time = None
            
            # Step 6: Run post-stop hooks
            await self._run_hooks(self._post_stop_hooks, "post-stop")
            
            logger.info(f"Adapter {self.adapter_id} stopped successfully")
            
            # Publish stop event
            await self._publish_lifecycle_event("stopped")
            
        except Exception as e:
            self._state = AdapterState.ERROR
            self._health_status = AdapterHealthStatus.UNHEALTHY
            self._last_error = e
            
            logger.error(f"Failed to stop adapter {self.adapter_id}: {e}")
            
            # Publish error event
            await self._publish_lifecycle_event("stop_failed", {"error": str(e)})
            
            raise AdapterLifecycleError(
                f"Failed to stop adapter {self.adapter_id}",
                self.adapter_id,
                self._state,
                e
            )
    
    async def restart(self) -> None:
        """Restart the adapter."""
        logger.info(f"Restarting adapter {self.adapter_id}")
        await self.stop()
        await self.start()
    
    async def _validate_configuration_internal(self) -> None:
        """Internal configuration validation with error handling."""
        try:
            # Validate base configuration
            validation_errors = self.configuration.validate()
            if validation_errors:
                raise AdapterValidationError(
                    f"Configuration validation failed for adapter {self.adapter_id}",
                    self.adapter_id,
                    validation_errors
                )
            
            # Call subclass validation
            await self._validate_configuration()
            
        except AdapterValidationError:
            raise
        except Exception as e:
            raise AdapterValidationError(
                f"Configuration validation error for adapter {self.adapter_id}: {e}",
                self.adapter_id,
                [str(e)]
            )
    
    async def _initialize_resources_internal(self) -> None:
        """Internal resource initialization with error handling."""
        try:
            await self._initialize_resources()
        except Exception as e:
            logger.error(f"Resource initialization failed for adapter {self.adapter_id}: {e}")
            raise
    
    async def _start_processing_internal(self) -> None:
        """Internal processing start with error handling."""
        try:
            await self._start_processing()
        except Exception as e:
            logger.error(f"Processing start failed for adapter {self.adapter_id}: {e}")
            raise
    
    async def _stop_processing_internal(self) -> None:
        """Internal processing stop with error handling."""
        try:
            await self._stop_processing()
        except Exception as e:
            logger.error(f"Processing stop failed for adapter {self.adapter_id}: {e}")
            raise
    
    async def _cleanup_resources_internal(self) -> None:
        """Internal resource cleanup with error handling."""
        try:
            await self._cleanup_resources()
        except Exception as e:
            logger.error(f"Resource cleanup failed for adapter {self.adapter_id}: {e}")
            raise
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        if self._health_check_task:
            return
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.debug(f"Started health monitoring for adapter {self.adapter_id}")
    
    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.debug(f"Stopped health monitoring for adapter {self.adapter_id}")
    
    async def _health_check_loop(self) -> None:
        """Health check monitoring loop."""
        while self._state == AdapterState.RUNNING:
            try:
                await asyncio.sleep(self.configuration.health_check_interval)
                
                if self._state != AdapterState.RUNNING:
                    break
                
                # Perform health check
                previous_status = self._health_status
                self._health_status = await self._perform_health_check()
                self._last_health_check = datetime.utcnow()
                
                # Log status changes
                if previous_status != self._health_status:
                    logger.info(f"Adapter {self.adapter_id} health status changed: {previous_status} -> {self._health_status}")
                    await self._publish_lifecycle_event("health_status_changed", {
                        "previous_status": previous_status.value,
                        "current_status": self._health_status.value
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for adapter {self.adapter_id}: {e}")
                self._health_status = AdapterHealthStatus.UNHEALTHY
    
    async def _perform_health_check(self) -> AdapterHealthStatus:
        """
        Perform health check and return status.
        
        Default implementation checks basic adapter state.
        Subclasses can override for custom health checks.
        """
        try:
            # Check if adapter is in running state
            if self._state != AdapterState.RUNNING:
                return AdapterHealthStatus.UNHEALTHY
            
            # Check error rate
            if self._metrics.error_rate > 0.5:  # More than 50% errors
                return AdapterHealthStatus.DEGRADED
            elif self._metrics.error_rate > 0.8:  # More than 80% errors
                return AdapterHealthStatus.UNHEALTHY
            
            # Check last activity
            if (self._metrics.last_activity and 
                datetime.utcnow() - self._metrics.last_activity > timedelta(minutes=5)):
                return AdapterHealthStatus.DEGRADED
            
            # Call subclass health check
            return await self._check_health()
            
        except Exception as e:
            logger.error(f"Health check error for adapter {self.adapter_id}: {e}")
            return AdapterHealthStatus.UNHEALTHY
    
    async def _run_hooks(self, hooks: List[Callable], hook_type: str) -> None:
        """Run lifecycle hooks with error handling."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            except Exception as e:
                logger.error(f"Error in {hook_type} hook for adapter {self.adapter_id}: {e}")
                # Continue with other hooks
    
    async def _publish_lifecycle_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Publish lifecycle events to event bus."""
        if not self.event_bus:
            return
        
        try:
            # Create a generic event for adapter lifecycle
            # In a real implementation, you might want specific event types
            event_data = {
                "adapter_id": self.adapter_id,
                "adapter_type": self.adapter_type,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "state": self._state.value,
                "health_status": self._health_status.value
            }
            
            if data:
                event_data.update(data)
            
            # For now, we'll log the event
            # In task 5.2 and 5.3, we'll create specific event types
            logger.info(f"Adapter lifecycle event: {event_data}")
            
        except Exception as e:
            logger.error(f"Failed to publish lifecycle event for adapter {self.adapter_id}: {e}")
    
    def add_pre_start_hook(self, hook: Callable) -> None:
        """Add a pre-start hook."""
        self._pre_start_hooks.append(hook)
    
    def add_post_start_hook(self, hook: Callable) -> None:
        """Add a post-start hook."""
        self._post_start_hooks.append(hook)
    
    def add_pre_stop_hook(self, hook: Callable) -> None:
        """Add a pre-stop hook."""
        self._pre_stop_hooks.append(hook)
    
    def add_post_stop_hook(self, hook: Callable) -> None:
        """Add a post-stop hook."""
        self._post_stop_hooks.append(hook)
    
    def update_metrics(self, processed: int = 0, failed: int = 0) -> None:
        """Update adapter metrics."""
        self._metrics.processed_items += processed
        self._metrics.failed_items += failed
        self._metrics.last_activity = datetime.utcnow()
        
        # Calculate error rate
        total = self._metrics.processed_items + self._metrics.failed_items
        if total > 0:
            self._metrics.error_rate = self._metrics.failed_items / total
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            "adapter_id": self.adapter_id,
            "adapter_type": self.adapter_type,
            "state": self._state.value,
            "health_status": self._health_status.value,
            "enabled": self.configuration.enabled,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "metrics": {
                "processed_items": self._metrics.processed_items,
                "failed_items": self._metrics.failed_items,
                "success_rate": self._metrics.success_rate,
                "error_rate": self._metrics.error_rate,
                "uptime_seconds": self._metrics.uptime.total_seconds() if self._metrics.uptime else 0,
                "last_activity": self._metrics.last_activity.isoformat() if self._metrics.last_activity else None
            },
            "last_error": str(self._last_error) if self._last_error else None,
            "retry_count": self._retry_count
        }
    
    # Template method hooks - to be implemented by subclasses
    
    @abstractmethod
    async def _validate_configuration(self) -> None:
        """
        Validate adapter-specific configuration.
        
        Should raise AdapterValidationError if validation fails.
        """
        pass
    
    @abstractmethod
    async def _initialize_resources(self) -> None:
        """
        Initialize adapter-specific resources.
        
        Called during startup after configuration validation.
        """
        pass
    
    @abstractmethod
    async def _start_processing(self) -> None:
        """
        Start adapter-specific processing.
        
        Called after resource initialization.
        """
        pass
    
    @abstractmethod
    async def _stop_processing(self) -> None:
        """
        Stop adapter-specific processing.
        
        Called during shutdown before resource cleanup.
        """
        pass
    
    @abstractmethod
    async def _cleanup_resources(self) -> None:
        """
        Clean up adapter-specific resources.
        
        Called during shutdown after processing is stopped.
        """
        pass
    
    async def _check_health(self) -> AdapterHealthStatus:
        """
        Perform adapter-specific health check.
        
        Default implementation returns HEALTHY.
        Subclasses can override for custom health checks.
        
        Returns:
            AdapterHealthStatus: Current health status
        """
        return AdapterHealthStatus.HEALTHY