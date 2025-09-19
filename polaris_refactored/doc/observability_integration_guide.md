# POLARIS Observability Integration Guide

This guide provides comprehensive instructions for integrating and using the observability system in POLARIS, including logging, metrics, and distributed tracing.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Component Integration](#component-integration)
5. [Manual Instrumentation](#manual-instrumentation)
6. [Metrics Collection](#metrics-collection)
7. [Distributed Tracing](#distributed-tracing)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

## Overview

The POLARIS observability system provides three pillars of observability:

- **Logging**: Structured JSON logging with correlation IDs and context
- **Metrics**: Business and technical metrics with Prometheus compatibility
- **Tracing**: Distributed tracing across component boundaries

### Key Features

- **Automatic Instrumentation**: Components can be automatically instrumented with decorators
- **Correlation Tracking**: Automatic correlation ID propagation across components
- **Environment-Specific Configuration**: Different configurations for dev, staging, production
- **Integration Ready**: Works with external systems (Jaeger, Prometheus, ELK stack)
- **Performance Optimized**: Minimal overhead with configurable sampling

## Quick Start

### 1. Basic Setup

```python
from src.infrastructure.observability import (
    ObservabilityConfig, initialize_observability, 
    get_logger, get_metrics_collector, get_tracer
)

# Initialize observability
config = ObservabilityConfig(
    service_name="my-polaris-service",
    log_level=LogLevel.INFO,
    enable_json_logging=True
)

observability_manager = initialize_observability(config)
await observability_manager.initialize()

# Get observability components
logger = get_logger("my.component")
metrics = get_metrics_collector()
tracer = get_tracer()
```

### 2. Component Instrumentation

```python
from src.infrastructure.observability import observe_polaris_component

@observe_polaris_component("my_service", auto_trace=True, auto_metrics=True)
class MyService:
    def __init__(self):
        # Logger, metrics, and tracer are automatically available as:
        # self._logger, self._metrics, self._tracer
        pass
    
    async def process_data(self, data):
        # This method is automatically instrumented
        self._logger.info("Processing data", extra={"size": len(data)})
        # ... processing logic ...
        return result
```

### 3. Framework Integration

```python
from src.framework.polaris_framework import PolarisFramework
from src.infrastructure.observability.config_examples import create_development_config

# Create framework with observability
observability_config = create_development_config()
framework = PolarisFramework(
    # ... other parameters ...
    observability_config=observability_config
)

await framework.start()
```

## Configuration

### Environment-Specific Configurations

#### Development
```python
from src.infrastructure.observability.config_examples import create_development_config

config = create_development_config(
    service_name="polaris-dev",
    log_level=LogLevel.DEBUG
)
```

#### Production
```python
from src.infrastructure.observability.config_examples import create_production_config

config = create_production_config(
    service_name="polaris",
    log_file_path="/var/log/polaris/polaris.log",
    jaeger_endpoint="http://jaeger-collector:14268/api/traces"
)
```

#### Custom Configuration
```python
config = ObservabilityConfig(
    service_name="polaris-custom",
    log_level=LogLevel.INFO,
    enable_json_logging=True,
    log_file_path="/custom/path/polaris.log",
    enable_console_tracing=False,
    jaeger_endpoint="http://custom-jaeger:14268/api/traces",
    metrics_export_interval=120,
    enable_auto_instrumentation=True
)
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `service_name` | Service identifier for observability | "polaris" |
| `log_level` | Minimum log level | INFO |
| `enable_json_logging` | Use JSON format for logs | True |
| `log_file_path` | File path for logs (None = console only) | None |
| `enable_console_tracing` | Output traces to console | True |
| `jaeger_endpoint` | Jaeger collector endpoint | None |
| `metrics_export_interval` | Metrics export frequency (seconds) | 60 |
| `enable_auto_instrumentation` | Automatic method instrumentation | True |

## Component Integration

### Automatic Integration with Decorator

The `@observe_polaris_component` decorator provides automatic observability integration:

```python
@observe_polaris_component(
    component_name="world_model",  # Optional, defaults to class name
    auto_trace=True,               # Enable automatic tracing
    auto_metrics=True,             # Enable automatic metrics
    log_method_calls=False         # Log all method calls (use sparingly)
)
class MyWorldModel:
    def __init__(self):
        # Observability components are automatically available:
        # self._logger: PolarisLogger
        # self._metrics: PolarisMetricsCollector  
        # self._tracer: PolarisTracer
        pass
    
    async def update_state(self, state):
        # Automatically instrumented with:
        # - Distributed tracing
        # - Duration metrics
        # - Error handling and logging
        pass
```

### Manual Integration

For more control, integrate observability manually:

```python
from src.infrastructure.observability import get_logger, get_metrics_collector, get_tracer

class MyComponent:
    def __init__(self):
        self.logger = get_logger("polaris.my_component")
        self.metrics = get_metrics_collector()
        self.tracer = get_tracer()
    
    async def my_method(self):
        with self.tracer.trace_operation("my_operation") as span:
            self.logger.info("Starting operation")
            
            # Add custom tags and events
            span.add_tag("operation_type", "custom")
            span.add_event("operation_started")
            
            # Your logic here
            await self.do_work()
            
            span.add_event("operation_completed")
            self.logger.info("Operation completed")
```

## Manual Instrumentation

### Logging Best Practices

```python
# Use structured logging with extra context
logger.info("Processing telemetry", extra={
    "system_id": system_id,
    "event_type": event_type,
    "metrics_count": len(metrics)
})

# Include error context
try:
    await risky_operation()
except Exception as e:
    logger.error("Operation failed", extra={
        "operation": "risky_operation",
        "error_type": type(e).__name__,
        "system_id": system_id
    }, exc_info=e)
```

### Context Managers for Correlation

```python
# Correlation context for request tracking
with logger.correlation_context() as correlation_id:
    logger.info("Processing request", extra={"request_id": "req_123"})
    await process_request()

# Adaptation context for adaptation flows
with logger.adaptation_context("adapt_123", "system_456"):
    logger.info("Starting adaptation")
    await execute_adaptation()

# System context for system-specific operations
with logger.system_context("system_789"):
    logger.info("Monitoring system")
    await collect_metrics()
```

### Custom Tracing Decorators

```python
from src.infrastructure.observability import (
    trace_adaptation_flow, trace_telemetry_processing, 
    trace_world_model_operation, trace_connector_operation
)

class AdaptationController:
    @trace_adaptation_flow("reactive_adaptation")
    async def execute_reactive_adaptation(self, system_id, actions):
        # Automatically traced with adaptation-specific tags
        pass
    
    @trace_telemetry_processing("system_telemetry")
    async def process_telemetry(self, telemetry_event):
        # Automatically traced with telemetry-specific tags
        pass

class WorldModel:
    @trace_world_model_operation("state_prediction")
    async def predict_future_state(self, current_state):
        # Automatically traced with world model tags
        pass

class DatabaseConnector:
    @trace_connector_operation("database", "query")
    async def execute_query(self, query):
        # Automatically traced with connector-specific tags
        pass
```

## Metrics Collection

### Built-in Metrics

POLARIS provides comprehensive built-in metrics:

#### System Health Metrics
- `polaris_system_health_score`: Overall health score (0-1)
- `polaris_active_systems_count`: Number of active systems

#### Adaptation Metrics
- `polaris_adaptations_triggered_total`: Total adaptations triggered
- `polaris_adaptations_successful_total`: Successful adaptations
- `polaris_adaptations_failed_total`: Failed adaptations
- `polaris_adaptation_duration_seconds`: Adaptation execution time

#### Telemetry Metrics
- `polaris_telemetry_events_received_total`: Telemetry events processed
- `polaris_telemetry_processing_duration_seconds`: Processing time
- `polaris_telemetry_queue_size`: Current queue size

#### Infrastructure Metrics
- `polaris_message_bus_latency_seconds`: Message bus latency
- `polaris_circuit_breaker_state_changes_total`: Circuit breaker changes
- `polaris_retry_attempts_total`: Retry attempts

### Custom Metrics

```python
# Register custom metrics
metrics = get_metrics_collector()

# Counter for custom events
custom_counter = metrics.register_counter(
    "my_custom_events_total",
    "Total custom events processed",
    ["event_type", "status"]
)

# Gauge for current values
custom_gauge = metrics.register_gauge(
    "my_custom_value",
    "Current custom value",
    ["component"]
)

# Histogram for distributions
custom_histogram = metrics.register_histogram(
    "my_custom_duration_seconds",
    "Duration of custom operations",
    labels=["operation_type"]
)

# Use custom metrics
custom_counter.increment(labels={"event_type": "test", "status": "success"})
custom_gauge.set(42.0, labels={"component": "processor"})

# Time operations
with Timer(custom_histogram, labels={"operation_type": "processing"}):
    await do_processing()
```

### Convenience Methods

```python
metrics = get_metrics_collector()

# Built-in convenience methods
metrics.increment_adaptations_triggered("system_1", "scale_out", "high_cpu")
metrics.increment_adaptations_successful("system_1", "scale_out")
metrics.set_system_health_score("system_1", 0.95)

# Timing helpers
with metrics.time_adaptation("system_1", "scale_out"):
    await execute_adaptation()

with metrics.time_telemetry_processing("system_1"):
    await process_telemetry()
```

## Distributed Tracing

### Automatic Tracing

Components decorated with `@observe_polaris_component` get automatic tracing:

```python
@observe_polaris_component("my_service", auto_trace=True)
class MyService:
    async def method1(self):
        # Automatically creates span: "my_service.method1"
        await self.method2()
    
    async def method2(self):
        # Automatically creates child span: "my_service.method2"
        pass
```

### Manual Tracing

```python
tracer = get_tracer()

# Basic operation tracing
with tracer.trace_operation("custom_operation") as span:
    span.add_tag("operation_type", "custom")
    span.add_event("started")
    
    await do_work()
    
    span.add_event("completed")

# Specialized tracing contexts
with tracer.trace_adaptation_flow("adapt_123", "system_456") as span:
    # Adaptation-specific tracing
    await execute_adaptation()

with tracer.trace_telemetry_processing("system_456", "metrics") as span:
    # Telemetry-specific tracing
    await process_telemetry()

with tracer.trace_world_model_update("system_456", "statistical") as span:
    # World model-specific tracing
    await update_model()
```

### Trace Correlation

Traces are automatically correlated across component boundaries:

```python
# In Component A
with tracer.trace_operation("component_a_operation") as span:
    # Get current trace context
    context = tracer.get_current_context()
    
    # Pass context to another component
    await component_b.process_with_context(context, data)

# In Component B
async def process_with_context(self, context, data):
    # Inject context to continue the trace
    self.tracer.inject_context(context)
    
    with self.tracer.trace_operation("component_b_operation") as span:
        # This span will be a child of the Component A span
        await self.do_processing(data)
```

### Trace Export

Configure trace export to external systems:

```python
# Jaeger export
config = ObservabilityConfig(
    jaeger_endpoint="http://jaeger-collector:14268/api/traces"
)

# Console export (development)
config = ObservabilityConfig(
    enable_console_tracing=True
)
```

## Production Deployment

### Recommended Production Configuration

```python
from src.infrastructure.observability.config_examples import create_production_config

config = create_production_config(
    service_name="polaris",
    log_file_path="/var/log/polaris/polaris.log",
    jaeger_endpoint="http://jaeger-collector:14268/api/traces"
)
```

### Log Management

#### Log Rotation
```bash
# Configure logrotate for POLARIS logs
# /etc/logrotate.d/polaris
/var/log/polaris/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 polaris polaris
    postrotate
        systemctl reload polaris
    endscript
}
```

#### Centralized Logging
```yaml
# Fluentd configuration for log aggregation
<source>
  @type tail
  path /var/log/polaris/polaris.log
  pos_file /var/log/fluentd/polaris.log.pos
  tag polaris.application
  format json
</source>

<match polaris.**>
  @type elasticsearch
  host elasticsearch.example.com
  port 9200
  index_name polaris
  type_name application
</match>
```

### Metrics Export

#### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'polaris'
    static_configs:
      - targets: ['polaris:8080']
    scrape_interval: 30s
    metrics_path: /metrics
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "POLARIS Observability",
    "panels": [
      {
        "title": "System Health Scores",
        "type": "graph",
        "targets": [
          {
            "expr": "polaris_system_health_score",
            "legendFormat": "{{system_id}}"
          }
        ]
      },
      {
        "title": "Adaptation Rate",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(polaris_adaptations_triggered_total[5m])",
            "legendFormat": "{{system_id}} - {{adaptation_type}}"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: polaris
    rules:
      - alert: PolarisSystemUnhealthy
        expr: polaris_system_health_score < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "POLARIS system {{ $labels.system_id }} is unhealthy"
          
      - alert: PolarisHighAdaptationFailureRate
        expr: rate(polaris_adaptations_failed_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High adaptation failure rate for {{ $labels.system_id }}"
```

### Performance Tuning

#### High-Performance Configuration
```python
config = ObservabilityConfig(
    service_name="polaris-hpc",
    log_level=LogLevel.ERROR,  # Minimal logging
    enable_json_logging=True,
    log_file_path="/var/log/polaris/polaris.log",
    enable_console_tracing=False,
    jaeger_endpoint=None,  # Disable tracing for max performance
    metrics_export_interval=300,  # Less frequent export
    enable_auto_instrumentation=False  # Manual instrumentation only
)
```

#### Sampling Configuration
```python
# Configure trace sampling for high-volume systems
tracer = get_tracer()
tracer.set_sampling_rate(0.1)  # Sample 10% of traces
```

## Troubleshooting

### Common Issues

#### 1. Missing Observability Data

**Problem**: No logs, metrics, or traces appearing

**Solution**:
```python
# Ensure observability is initialized
observability_manager = initialize_observability(config)
await observability_manager.initialize()

# Check configuration
print(f"Service name: {config.service_name}")
print(f"Log level: {config.log_level}")
print(f"Exporters configured: {len(tracer.exporters)}")
```

#### 2. High Memory Usage

**Problem**: Observability system consuming too much memory

**Solution**:
```python
# Reduce metrics export interval
config.metrics_export_interval = 300  # 5 minutes

# Disable auto-instrumentation for high-volume methods
config.enable_auto_instrumentation = False

# Use manual instrumentation selectively
@observe_polaris_component("service", auto_trace=False, auto_metrics=False)
class MyService:
    pass
```

#### 3. Performance Impact

**Problem**: Observability causing performance degradation

**Solution**:
```python
# Use higher log level
config.log_level = LogLevel.WARNING

# Disable console tracing
config.enable_console_tracing = False

# Reduce trace sampling
tracer.set_sampling_rate(0.01)  # 1% sampling
```

#### 4. Correlation ID Issues

**Problem**: Correlation IDs not propagating correctly

**Solution**:
```python
# Ensure context managers are used properly
with logger.correlation_context() as correlation_id:
    # All operations within this block will have the same correlation ID
    await operation1()
    await operation2()

# For async operations, ensure context is preserved
import contextvars
correlation_id = contextvars.copy_context()
await asyncio.create_task(operation(), context=correlation_id)
```

### Debugging Observability

#### Enable Debug Logging
```python
import logging
logging.getLogger("polaris.observability").setLevel(logging.DEBUG)
```

#### Check Metrics Collection
```python
metrics = get_metrics_collector()
all_metrics = metrics.get_all_metrics()
for name, metric in all_metrics.items():
    print(f"{name}: {metric.get_value()}")
```

#### Verify Trace Export
```python
tracer = get_tracer()
tracer.flush()  # Force export of pending traces
```

### Performance Monitoring

Monitor the observability system itself:

```python
# Add observability metrics
observability_metrics = metrics.register_histogram(
    "polaris_observability_overhead_seconds",
    "Time spent on observability operations",
    labels=["operation_type"]
)

# Monitor logging overhead
with Timer(observability_metrics, labels={"operation_type": "logging"}):
    logger.info("Test message")

# Monitor metrics collection overhead  
with Timer(observability_metrics, labels={"operation_type": "metrics"}):
    metrics.increment_adaptations_triggered("test", "test", "test")
```

## Best Practices

1. **Use Structured Logging**: Always include relevant context in log messages
2. **Correlation IDs**: Use correlation contexts for request tracking
3. **Appropriate Log Levels**: Use DEBUG sparingly, INFO for important events
4. **Metric Naming**: Follow Prometheus naming conventions
5. **Trace Sampling**: Use sampling in high-volume production environments
6. **Error Handling**: Always log errors with full context
7. **Performance**: Monitor observability overhead and tune accordingly
8. **Security**: Avoid logging sensitive information
9. **Retention**: Configure appropriate log and metric retention policies
10. **Testing**: Include observability in your testing strategy

## Integration Examples

See `examples/observability_integration_example.py` for comprehensive examples of:
- Basic setup and configuration
- Component instrumentation
- Manual tracing and correlation
- Environment-specific configurations
- Metrics collection and analysis
- Error handling with observability