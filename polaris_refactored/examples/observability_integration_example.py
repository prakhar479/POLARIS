"""
POLARIS Observability Integration Example

This example demonstrates how to set up and use the comprehensive observability
system in POLARIS, including logging, metrics, and tracing across all components.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from src.infrastructure.observability import (
    ObservabilityConfig, initialize_observability, shutdown_observability,
    get_logger, get_metrics_collector, get_tracer, observe_polaris_component,
    trace_adaptation_flow, trace_telemetry_processing, trace_world_model_operation
)
from src.infrastructure.observability.config_examples import (
    create_development_config, create_production_config, create_observability_config
)
from src.domain.models import SystemState, MetricValue, HealthStatus
from src.framework.events import TelemetryEvent


# Example 1: Basic observability setup
async def basic_observability_example():
    """Demonstrate basic observability setup and usage."""
    print("=== Basic Observability Example ===")
    
    # Initialize observability with development configuration
    config = create_development_config(service_name="polaris-example")
    observability_manager = initialize_observability(config)
    await observability_manager.initialize()
    
    # Get observability components
    logger = get_logger("example.basic")
    metrics = get_metrics_collector()
    tracer = get_tracer()
    
    # Demonstrate logging
    logger.info("Starting basic observability example", extra={
        "example_type": "basic",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Demonstrate metrics
    metrics.increment_adaptations_triggered("example_system", "manual", "demonstration")
    metrics.set_system_health_score("example_system", 0.95)
    
    # Demonstrate tracing
    with tracer.trace_operation("example_operation") as span:
        span.add_tag("operation_type", "demonstration")
        span.add_event("operation_started")
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        span.add_event("operation_completed")
    
    logger.info("Basic observability example completed")
    
    # Cleanup
    await shutdown_observability()


# Example 2: Component instrumentation
@observe_polaris_component("example_service", auto_trace=True, auto_metrics=True, log_method_calls=True)
class ExampleService:
    """Example service with automatic observability instrumentation."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        # Logger, metrics, and tracer are automatically added by the decorator
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with automatic observability."""
        # This method is automatically instrumented with:
        # - Logging of method calls
        # - Metrics collection (duration, success/failure)
        # - Distributed tracing
        
        self._logger.info("Processing data", extra={
            "data_size": len(data),
            "service_name": self.service_name
        })
        
        # Simulate processing
        await asyncio.sleep(0.05)
        
        result = {"processed": True, "input_size": len(data)}
        
        self._logger.debug("Data processing completed", extra={
            "result": result
        })
        
        return result
    
    @trace_telemetry_processing("custom_telemetry")
    async def handle_telemetry(self, telemetry: TelemetryEvent) -> None:
        """Handle telemetry with custom tracing."""
        self._logger.info("Handling telemetry event", extra={
            "system_id": telemetry.system_state.system_id,
            "health_status": telemetry.system_state.health_status.value
        })
        
        # Update metrics
        self._metrics.increment_telemetry_events_received(
            telemetry.system_state.system_id,
            "custom_event"
        )
        
        # Simulate telemetry processing
        await asyncio.sleep(0.02)


async def component_instrumentation_example():
    """Demonstrate automatic component instrumentation."""
    print("\n=== Component Instrumentation Example ===")
    
    # Initialize observability
    config = create_development_config(service_name="polaris-instrumentation")
    observability_manager = initialize_observability(config)
    await observability_manager.initialize()
    
    # Create instrumented service
    service = ExampleService("example_service")
    
    # Use the service - all methods are automatically instrumented
    test_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
    result = await service.process_data(test_data)
    print(f"Processing result: {result}")
    
    # Create and handle telemetry
    system_state = SystemState(
        system_id="example_system",
        timestamp=datetime.utcnow(),
        metrics={
            "cpu_usage": MetricValue("cpu_usage", 0.75, "ratio"),
            "memory_usage": MetricValue("memory_usage", 0.60, "ratio")
        },
        health_status=HealthStatus.HEALTHY
    )
    
    telemetry = TelemetryEvent(system_state)
    await service.handle_telemetry(telemetry)
    
    # Cleanup
    await shutdown_observability()


# Example 3: Manual tracing and correlation
async def manual_tracing_example():
    """Demonstrate manual tracing with correlation."""
    print("\n=== Manual Tracing Example ===")
    
    # Initialize observability
    config = create_development_config(service_name="polaris-tracing")
    observability_manager = initialize_observability(config)
    await observability_manager.initialize()
    
    logger = get_logger("example.tracing")
    tracer = get_tracer()
    
    # Start a trace for an adaptation flow
    with tracer.trace_adaptation_flow("example_adaptation", "example_system") as span:
        logger.info("Starting adaptation flow", extra={
            "adaptation_id": "example_adaptation",
            "system_id": "example_system"
        })
        
        # Simulate monitoring phase
        with tracer.trace_operation("monitor_phase") as monitor_span:
            monitor_span.add_tag("phase", "monitor")
            logger.debug("Monitoring system state")
            await asyncio.sleep(0.02)
        
        # Simulate analysis phase
        with tracer.trace_operation("analyze_phase") as analyze_span:
            analyze_span.add_tag("phase", "analyze")
            analyze_span.add_event("analysis_started")
            logger.debug("Analyzing adaptation need")
            await asyncio.sleep(0.03)
            analyze_span.add_event("analysis_completed", {"result": "adaptation_needed"})
        
        # Simulate planning phase
        with tracer.trace_operation("plan_phase") as plan_span:
            plan_span.add_tag("phase", "plan")
            logger.debug("Planning adaptation actions")
            await asyncio.sleep(0.02)
            plan_span.add_tag("actions_planned", 2)
        
        # Simulate execution phase
        with tracer.trace_operation("execute_phase") as execute_span:
            execute_span.add_tag("phase", "execute")
            logger.info("Executing adaptation actions")
            await asyncio.sleep(0.05)
            execute_span.add_tag("execution_status", "success")
        
        span.add_event("adaptation_completed")
        logger.info("Adaptation flow completed successfully")
    
    # Cleanup
    await shutdown_observability()


# Example 4: Environment-specific configuration
async def environment_configuration_example():
    """Demonstrate different observability configurations for different environments."""
    print("\n=== Environment Configuration Example ===")
    
    environments = ["development", "testing", "staging", "production"]
    
    for env in environments:
        print(f"\n--- {env.upper()} Configuration ---")
        
        # Create environment-specific configuration
        config = create_observability_config(environment=env)
        
        print(f"Service Name: {config.service_name}")
        print(f"Log Level: {config.log_level.value}")
        print(f"JSON Logging: {config.enable_json_logging}")
        print(f"Log File: {config.log_file_path}")
        print(f"Console Tracing: {config.enable_console_tracing}")
        print(f"Jaeger Endpoint: {config.jaeger_endpoint}")
        print(f"Metrics Export Interval: {config.metrics_export_interval}s")
        print(f"Auto Instrumentation: {config.enable_auto_instrumentation}")


# Example 5: Metrics collection and analysis
async def metrics_example():
    """Demonstrate comprehensive metrics collection."""
    print("\n=== Metrics Collection Example ===")
    
    # Initialize observability
    config = create_development_config(service_name="polaris-metrics")
    observability_manager = initialize_observability(config)
    await observability_manager.initialize()
    
    logger = get_logger("example.metrics")
    metrics = get_metrics_collector()
    
    # Simulate system activity with metrics
    systems = ["web_server", "database", "cache", "api_gateway"]
    
    for system_id in systems:
        logger.info(f"Simulating activity for {system_id}")
        
        # Simulate successful adaptations
        for i in range(3):
            with metrics.time_adaptation(system_id, "scale_out"):
                await asyncio.sleep(0.01)  # Simulate adaptation time
            
            metrics.increment_adaptations_successful(system_id, "scale_out")
        
        # Simulate some failures
        metrics.increment_adaptations_failed(system_id, "scale_out", "resource_limit")
        
        # Set health scores
        import random
        health_score = random.uniform(0.7, 1.0)
        metrics.set_system_health_score(system_id, health_score)
        
        # Simulate telemetry processing
        for j in range(5):
            with metrics.time_telemetry_processing(system_id):
                await asyncio.sleep(0.005)  # Simulate processing time
    
    # Update overall system count
    metrics.set_active_systems_count(len(systems))
    
    # Display some metrics
    print("\nCollected Metrics:")
    all_metrics = metrics.get_all_metrics()
    for name, metric in all_metrics.items():
        if "polaris_system_health_score" in name:
            for system_id in systems:
                value = metric.get_value({"system_id": system_id})
                print(f"  {system_id} health score: {value.value:.3f}")
    
    # Cleanup
    await shutdown_observability()


# Example 6: Error handling and observability
async def error_handling_example():
    """Demonstrate error handling with observability."""
    print("\n=== Error Handling Example ===")
    
    # Initialize observability
    config = create_development_config(service_name="polaris-errors")
    observability_manager = initialize_observability(config)
    await observability_manager.initialize()
    
    logger = get_logger("example.errors")
    tracer = get_tracer()
    
    # Simulate error scenarios with proper observability
    with tracer.trace_operation("error_simulation") as span:
        try:
            logger.info("Starting operation that will fail")
            
            # Simulate some work before failure
            await asyncio.sleep(0.01)
            
            # Simulate an error
            raise ValueError("Simulated error for demonstration")
            
        except ValueError as e:
            # Error is automatically captured by the span
            logger.error("Operation failed", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, exc_info=e)
            
            # The span will automatically be marked as error
            span.add_event("error_handled")
    
    logger.info("Error handling demonstration completed")
    
    # Cleanup
    await shutdown_observability()


# Main example runner
async def main():
    """Run all observability examples."""
    print("POLARIS Observability Integration Examples")
    print("=" * 50)
    
    try:
        await basic_observability_example()
        await component_instrumentation_example()
        await manual_tracing_example()
        await environment_configuration_example()
        await metrics_example()
        await error_handling_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())