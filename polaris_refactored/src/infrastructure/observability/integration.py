"""
Observability Integration Module

This module provides comprehensive integration of logging, metrics, and tracing
across all POLARIS components. It includes automatic instrumentation, correlation
tracking, and centralized observability configuration.
"""

import asyncio
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime
import inspect

from .logging import PolarisLogger, get_logger, LogLevel, configure_default_logging
from .metrics import PolarisMetricsCollector, get_metrics_collector, Timer
from .tracing import PolarisTracer, get_tracer, SpanKind, configure_tracing, trace_polaris_method
from ..di import Injectable
from ..exceptions import PolarisException

F = TypeVar('F', bound=Callable[..., Any])


class ObservabilityConfig:
    """Configuration for observability integration."""
    
    def __init__(
        self,
        service_name: str = "polaris",
        log_level: LogLevel = LogLevel.INFO,
        enable_json_logging: bool = True,
        log_file_path: Optional[str] = None,
        enable_console_tracing: bool = True,
        jaeger_endpoint: Optional[str] = None,
        metrics_export_interval: int = 60,
        enable_auto_instrumentation: bool = True
    ):
        self.service_name = service_name
        self.log_level = log_level
        self.enable_json_logging = enable_json_logging
        self.log_file_path = log_file_path
        self.enable_console_tracing = enable_console_tracing
        self.jaeger_endpoint = jaeger_endpoint
        self.metrics_export_interval = metrics_export_interval
        self.enable_auto_instrumentation = enable_auto_instrumentation


class ObservabilityManager(Injectable):
    """
    Central manager for all observability concerns in POLARIS.
    
    Provides unified configuration and management of logging, metrics, and tracing
    across all system components.
    """
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.logger = get_logger("polaris.observability")
        self.metrics_collector = get_metrics_collector()
        self.tracer = get_tracer()
        self._initialized = False
        self._metrics_export_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize observability systems."""
        if self._initialized:
            return
        
        # Configure logging
        configure_default_logging(
            level=self.config.log_level,
            use_json=self.config.enable_json_logging,
            log_file=self.config.log_file_path
        )
        
        # Configure tracing
        configure_tracing(
            service_name=self.config.service_name,
            console_export=self.config.enable_console_tracing,
            jaeger_endpoint=self.config.jaeger_endpoint
        )
        
        # Start metrics export task
        if self.config.metrics_export_interval > 0:
            self._metrics_export_task = asyncio.create_task(
                self._metrics_export_loop()
            )
        
        self._initialized = True
        self.logger.info("Observability manager initialized", extra={
            "service_name": self.config.service_name,
            "log_level": self.config.log_level.value,
            "tracing_enabled": True,
            "metrics_enabled": True
        })
    
    async def shutdown(self) -> None:
        """Shutdown observability systems."""
        if not self._initialized:
            return
        
        self.logger.info("Shutting down observability manager")
        
        # Stop metrics export task
        if self._metrics_export_task:
            self._metrics_export_task.cancel()
            try:
                await self._metrics_export_task
            except asyncio.CancelledError:
                pass
        
        # Flush any pending traces
        self.tracer.flush()
        
        self._initialized = False
    
    async def _metrics_export_loop(self) -> None:
        """Periodic metrics export loop."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_export_interval)
                
                # Export metrics (in a real implementation, this would send to external systems)
                metrics = self.metrics_collector.get_all_metrics()
                self.logger.debug(f"Exported {len(metrics)} metrics")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in metrics export loop", extra={"error": str(e)})
                await asyncio.sleep(5)  # Brief pause before retrying
    
    def get_logger(self, name: str) -> PolarisLogger:
        """Get a logger instance with observability integration."""
        return get_logger(name)
    
    def get_metrics_collector(self) -> PolarisMetricsCollector:
        """Get the metrics collector instance."""
        return self.metrics_collector
    
    def get_tracer(self) -> PolarisTracer:
        """Get the tracer instance."""
        return self.tracer


def observe_polaris_component(
    component_name: Optional[str] = None,
    auto_trace: bool = True,
    auto_metrics: bool = True,
    log_method_calls: bool = False
):
    """
    Class decorator for automatic observability instrumentation of POLARIS components.
    
    Usage:
        @observe_polaris_component("world_model", auto_trace=True, auto_metrics=True)
        class PolarisWorldModel:
            pass
    """
    def decorator(cls):
        # Get component name
        comp_name = component_name or cls.__name__.lower()
        
        # Add observability attributes
        cls._observability_component_name = comp_name
        cls._observability_auto_trace = auto_trace
        cls._observability_auto_metrics = auto_metrics
        cls._observability_log_calls = log_method_calls
        
        # Get logger for this component
        logger = get_logger(f"polaris.{comp_name}")
        cls._logger = logger
        
        # Get metrics collector
        metrics = get_metrics_collector()
        cls._metrics = metrics
        
        # Get tracer
        tracer = get_tracer()
        cls._tracer = tracer
        
        # Auto-instrument methods if enabled
        if auto_trace or auto_metrics or log_method_calls:
            _instrument_class_methods(cls, comp_name, auto_trace, auto_metrics, log_method_calls)
        
        return cls
    
    return decorator


def _instrument_class_methods(
    cls, 
    component_name: str, 
    auto_trace: bool, 
    auto_metrics: bool, 
    log_calls: bool
):
    """Automatically instrument class methods with observability."""
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        
        # Skip private methods, properties, and non-callable attributes
        if (attr_name.startswith('_') or 
            not callable(attr) or 
            isinstance(attr, (property, staticmethod, classmethod))):
            continue
        
        # Skip if already instrumented
        if hasattr(attr, '_polaris_instrumented'):
            continue
        
        # Create instrumented version
        instrumented_method = _create_instrumented_method(
            attr, component_name, attr_name, auto_trace, auto_metrics, log_calls
        )
        
        # Mark as instrumented
        instrumented_method._polaris_instrumented = True
        
        # Replace the method
        setattr(cls, attr_name, instrumented_method)


def _create_instrumented_method(
    method: Callable,
    component_name: str,
    method_name: str,
    auto_trace: bool,
    auto_metrics: bool,
    log_calls: bool
) -> Callable:
    """Create an instrumented version of a method."""
    
    @functools.wraps(method)
    def sync_wrapper(self, *args, **kwargs):
        return _execute_with_observability(
            method, self, args, kwargs, component_name, method_name,
            auto_trace, auto_metrics, log_calls, is_async=False
        )
    
    @functools.wraps(method)
    async def async_wrapper(self, *args, **kwargs):
        return await _execute_with_observability(
            method, self, args, kwargs, component_name, method_name,
            auto_trace, auto_metrics, log_calls, is_async=True
        )
    
    # Return appropriate wrapper
    if asyncio.iscoroutinefunction(method):
        return async_wrapper
    else:
        return sync_wrapper


def _execute_with_observability(
    method: Callable,
    instance: Any,
    args: tuple,
    kwargs: dict,
    component_name: str,
    method_name: str,
    auto_trace: bool,
    auto_metrics: bool,
    log_calls: bool,
    is_async: bool
):
    """Execute method with observability instrumentation."""
    
    # Get observability instances
    logger = getattr(instance, '_logger', get_logger(f"polaris.{component_name}"))
    metrics = getattr(instance, '_metrics', get_metrics_collector())
    tracer = getattr(instance, '_tracer', get_tracer())
    
    operation_name = f"{component_name}.{method_name}"
    
    # Log method call if enabled
    if log_calls:
        logger.debug(f"Calling {operation_name}", extra={
            "component": component_name,
            "method": method_name,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        })
    
    # Create metrics timer if enabled
    timer = None
    if auto_metrics:
        # Create method-specific histogram if it doesn't exist
        histogram_name = f"polaris_{component_name}_method_duration_seconds"
        histogram = metrics.get_metric(histogram_name)
        if not histogram:
            histogram = metrics.register_histogram(
                histogram_name,
                f"Duration of {component_name} method calls",
                labels=["method"]
            )
        
        if histogram:
            timer = Timer(histogram, labels={"method": method_name})
    
    # Execute with tracing if enabled
    if auto_trace:
        tags = {
            "component": component_name,
            "method": method_name,
            "class": instance.__class__.__name__
        }
        
        if is_async:
            return _execute_async_with_trace_and_metrics(
                method, instance, args, kwargs, tracer, operation_name, tags, timer, logger
            )
        else:
            return _execute_sync_with_trace_and_metrics(
                method, instance, args, kwargs, tracer, operation_name, tags, timer, logger
            )
    else:
        # Execute without tracing but with metrics
        if is_async:
            return _execute_async_with_metrics(method, instance, args, kwargs, timer, logger)
        else:
            return _execute_sync_with_metrics(method, instance, args, kwargs, timer, logger)


def _execute_sync_with_trace_and_metrics(
    method: Callable, instance: Any, args: tuple, kwargs: dict,
    tracer: PolarisTracer, operation_name: str, tags: dict,
    timer: Optional[Timer], logger: PolarisLogger
):
    """Execute synchronous method with tracing and metrics."""
    with tracer.trace_operation(operation_name, SpanKind.INTERNAL, tags) as span:
        if timer:
            with timer:
                try:
                    result = method(instance, *args, **kwargs)
                    span.add_tag("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    logger.error(f"Error in {operation_name}", extra={"error": str(e)}, exc_info=e)
                    raise
        else:
            try:
                result = method(instance, *args, **kwargs)
                span.add_tag("success", True)
                return result
            except Exception as e:
                span.set_error(e)
                logger.error(f"Error in {operation_name}", extra={"error": str(e)}, exc_info=e)
                raise


async def _execute_async_with_trace_and_metrics(
    method: Callable, instance: Any, args: tuple, kwargs: dict,
    tracer: PolarisTracer, operation_name: str, tags: dict,
    timer: Optional[Timer], logger: PolarisLogger
):
    """Execute asynchronous method with tracing and metrics."""
    with tracer.trace_operation(operation_name, SpanKind.INTERNAL, tags) as span:
        if timer:
            with timer:
                try:
                    result = await method(instance, *args, **kwargs)
                    span.add_tag("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    logger.error(f"Error in {operation_name}", extra={"error": str(e)}, exc_info=e)
                    raise
        else:
            try:
                result = await method(instance, *args, **kwargs)
                span.add_tag("success", True)
                return result
            except Exception as e:
                span.set_error(e)
                logger.error(f"Error in {operation_name}", extra={"error": str(e)}, exc_info=e)
                raise


def _execute_sync_with_metrics(
    method: Callable, instance: Any, args: tuple, kwargs: dict,
    timer: Optional[Timer], logger: PolarisLogger
):
    """Execute synchronous method with metrics only."""
    if timer:
        with timer:
            return method(instance, *args, **kwargs)
    else:
        return method(instance, *args, **kwargs)


async def _execute_async_with_metrics(
    method: Callable, instance: Any, args: tuple, kwargs: dict,
    timer: Optional[Timer], logger: PolarisLogger
):
    """Execute asynchronous method with metrics only."""
    if timer:
        with timer:
            return await method(instance, *args, **kwargs)
    else:
        return await method(instance, *args, **kwargs)


# Convenience decorators for specific observability patterns
def trace_adaptation_flow(adaptation_type: str):
    """Decorator for tracing adaptation flows."""
    def decorator(func: F) -> F:
        return trace_polaris_method(
            operation_name=f"adaptation.{adaptation_type}",
            kind=SpanKind.INTERNAL,
            tags={"adaptation_type": adaptation_type, "component": "adaptation"}
        )(func)
    return decorator


def trace_telemetry_processing(event_type: str):
    """Decorator for tracing telemetry processing."""
    def decorator(func: F) -> F:
        return trace_polaris_method(
            operation_name=f"telemetry.{event_type}",
            kind=SpanKind.CONSUMER,
            tags={"event_type": event_type, "component": "telemetry"}
        )(func)
    return decorator


def trace_world_model_operation(operation: str):
    """Decorator for tracing world model operations."""
    def decorator(func: F) -> F:
        return trace_polaris_method(
            operation_name=f"world_model.{operation}",
            kind=SpanKind.INTERNAL,
            tags={"operation": operation, "component": "world_model"}
        )(func)
    return decorator


def trace_connector_operation(connector_type: str, operation: str):
    """Decorator for tracing connector operations."""
    def decorator(func: F) -> F:
        return trace_polaris_method(
            operation_name=f"connector.{connector_type}.{operation}",
            kind=SpanKind.CLIENT,
            tags={"connector_type": connector_type, "operation": operation, "component": "connector"}
        )(func)
    return decorator


# Global observability manager instance
_observability_manager: Optional[ObservabilityManager] = None


def get_observability_manager() -> Optional[ObservabilityManager]:
    """Get the global observability manager instance."""
    return _observability_manager


def initialize_observability(config: ObservabilityConfig) -> ObservabilityManager:
    """Initialize global observability manager."""
    global _observability_manager
    _observability_manager = ObservabilityManager(config)
    return _observability_manager


async def shutdown_observability() -> None:
    """Shutdown global observability manager."""
    global _observability_manager
    if _observability_manager:
        await _observability_manager.shutdown()
        _observability_manager = None