"""
Observability Configuration Examples

This module provides example configurations for different deployment scenarios
and demonstrates how to set up comprehensive observability for POLARIS.
"""

from pathlib import Path
from typing import Optional

from .integration import ObservabilityConfig
from .logging import LogLevel


def create_development_config(
    service_name: str = "polaris-dev",
    log_level: LogLevel = LogLevel.DEBUG
) -> ObservabilityConfig:
    """
    Create observability configuration for development environment.
    
    Features:
    - Debug level logging
    - Human-readable console output
    - Console tracing for immediate feedback
    - Frequent metrics export
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=log_level,
        enable_json_logging=False,  # Human-readable for development
        log_file_path=None,  # Console only
        enable_console_tracing=True,
        jaeger_endpoint=None,  # No external tracing in dev
        metrics_export_interval=30,  # Frequent export for development
        enable_auto_instrumentation=True
    )


def create_testing_config(
    service_name: str = "polaris-test",
    log_file_path: Optional[str] = "./logs/polaris-test.log"
) -> ObservabilityConfig:
    """
    Create observability configuration for testing environment.
    
    Features:
    - Info level logging
    - JSON logging for structured analysis
    - File logging for test analysis
    - Console tracing for debugging
    - Moderate metrics export frequency
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=LogLevel.INFO,
        enable_json_logging=True,
        log_file_path=log_file_path,
        enable_console_tracing=True,
        jaeger_endpoint=None,  # Could be added for integration tests
        metrics_export_interval=60,
        enable_auto_instrumentation=True
    )


def create_staging_config(
    service_name: str = "polaris-staging",
    log_file_path: str = "/var/log/polaris/polaris-staging.log",
    jaeger_endpoint: Optional[str] = "http://jaeger-collector:14268/api/traces"
) -> ObservabilityConfig:
    """
    Create observability configuration for staging environment.
    
    Features:
    - Info level logging
    - JSON logging for log aggregation
    - File logging with rotation
    - External tracing system integration
    - Standard metrics export frequency
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=LogLevel.INFO,
        enable_json_logging=True,
        log_file_path=log_file_path,
        enable_console_tracing=False,  # Reduce noise in staging
        jaeger_endpoint=jaeger_endpoint,
        metrics_export_interval=60,
        enable_auto_instrumentation=True
    )


def create_production_config(
    service_name: str = "polaris",
    log_file_path: str = "/var/log/polaris/polaris.log",
    jaeger_endpoint: str = "http://jaeger-collector:14268/api/traces"
) -> ObservabilityConfig:
    """
    Create observability configuration for production environment.
    
    Features:
    - Warning level logging (reduce noise)
    - JSON logging for log aggregation systems
    - File logging with proper rotation
    - External tracing system integration
    - Optimized metrics export frequency
    - Full auto-instrumentation
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=LogLevel.WARNING,  # Reduce log volume in production
        enable_json_logging=True,
        log_file_path=log_file_path,
        enable_console_tracing=False,  # No console output in production
        jaeger_endpoint=jaeger_endpoint,
        metrics_export_interval=120,  # Less frequent to reduce overhead
        enable_auto_instrumentation=True
    )


def create_high_performance_config(
    service_name: str = "polaris-hpc",
    log_file_path: str = "/var/log/polaris/polaris-hpc.log",
    jaeger_endpoint: Optional[str] = None
) -> ObservabilityConfig:
    """
    Create observability configuration for high-performance environments.
    
    Features:
    - Error level logging only
    - JSON logging for efficiency
    - File logging only
    - Minimal tracing overhead
    - Reduced metrics export frequency
    - Selective auto-instrumentation
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=LogLevel.ERROR,  # Minimal logging
        enable_json_logging=True,
        log_file_path=log_file_path,
        enable_console_tracing=False,
        jaeger_endpoint=jaeger_endpoint,  # Optional for performance
        metrics_export_interval=300,  # Less frequent export
        enable_auto_instrumentation=False  # Manual instrumentation only
    )


def create_debugging_config(
    service_name: str = "polaris-debug",
    log_file_path: str = "./logs/polaris-debug.log"
) -> ObservabilityConfig:
    """
    Create observability configuration for debugging scenarios.
    
    Features:
    - Debug level logging
    - Human-readable console output
    - Detailed file logging
    - Console tracing for immediate feedback
    - Frequent metrics export
    - Full instrumentation
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=LogLevel.DEBUG,
        enable_json_logging=False,  # Human-readable for debugging
        log_file_path=log_file_path,
        enable_console_tracing=True,
        jaeger_endpoint=None,
        metrics_export_interval=15,  # Very frequent for debugging
        enable_auto_instrumentation=True
    )


def create_cloud_native_config(
    service_name: str = "polaris",
    jaeger_endpoint: str = "http://jaeger-collector.observability.svc.cluster.local:14268/api/traces"
) -> ObservabilityConfig:
    """
    Create observability configuration for cloud-native/Kubernetes environments.
    
    Features:
    - Info level logging
    - JSON logging for log aggregation (ELK, Fluentd, etc.)
    - No file logging (stdout/stderr only)
    - External tracing system integration
    - Standard metrics export (Prometheus scraping)
    - Full auto-instrumentation
    """
    return ObservabilityConfig(
        service_name=service_name,
        log_level=LogLevel.INFO,
        enable_json_logging=True,
        log_file_path=None,  # Use stdout/stderr for container logs
        enable_console_tracing=False,
        jaeger_endpoint=jaeger_endpoint,
        metrics_export_interval=60,
        enable_auto_instrumentation=True
    )


# Configuration factory function
def create_observability_config(
    environment: str = "development",
    service_name: Optional[str] = None,
    **kwargs
) -> ObservabilityConfig:
    """
    Factory function to create observability configuration based on environment.
    
    Args:
        environment: Target environment (development, testing, staging, production, etc.)
        service_name: Optional service name override
        **kwargs: Additional configuration overrides
    
    Returns:
        ObservabilityConfig: Configured observability settings
    """
    # Default service names by environment
    default_service_names = {
        "development": "polaris-dev",
        "testing": "polaris-test", 
        "staging": "polaris-staging",
        "production": "polaris",
        "high_performance": "polaris-hpc",
        "debugging": "polaris-debug",
        "cloud_native": "polaris"
    }
    
    # Use provided service name or default
    if service_name is None:
        service_name = default_service_names.get(environment, "polaris")
    
    # Create base configuration
    if environment == "development":
        config = create_development_config(service_name)
    elif environment == "testing":
        config = create_testing_config(service_name)
    elif environment == "staging":
        config = create_staging_config(service_name)
    elif environment == "production":
        config = create_production_config(service_name)
    elif environment == "high_performance":
        config = create_high_performance_config(service_name)
    elif environment == "debugging":
        config = create_debugging_config(service_name)
    elif environment == "cloud_native":
        config = create_cloud_native_config(service_name)
    else:
        # Default to development configuration
        config = create_development_config(service_name)
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Example usage and integration patterns
def setup_observability_for_framework(
    environment: str = "development",
    custom_config: Optional[ObservabilityConfig] = None
) -> ObservabilityConfig:
    """
    Set up observability for the POLARIS framework.
    
    This function demonstrates the recommended pattern for integrating
    observability into the POLARIS framework initialization.
    """
    if custom_config:
        config = custom_config
    else:
        config = create_observability_config(environment)
    
    # Ensure log directory exists if file logging is enabled
    if config.log_file_path:
        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    return config


# Environment-specific examples
DEVELOPMENT_CONFIG = create_development_config()
TESTING_CONFIG = create_testing_config()
STAGING_CONFIG = create_staging_config()
PRODUCTION_CONFIG = create_production_config()
CLOUD_NATIVE_CONFIG = create_cloud_native_config()