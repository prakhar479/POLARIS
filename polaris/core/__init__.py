"""Polaris core package."""

from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.component_builder import ComponentBuilder
from polaris.core.config_reloader import ConfigReloader
from polaris.core.events import AdaptationEvent, EventBus, TelemetryEvent
from polaris.core.meta_learning_loop import MetaLearningLoop
from polaris.core.metrics_export_loop import MetricsExportLoop
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.core.monitoring_loop import MonitoringLoop
from polaris.core.polaris import Polaris, PolarisConfig
from polaris.core.registry import ConnectorRegistry

__all__ = [
    # Models
    "MetricValue",
    "SystemState",
    "AdaptationAction",
    "ExecutionResult",
    "HealthStatus",
    "ExecutionStatus",
    # Events
    "EventBus",
    "TelemetryEvent",
    "AdaptationEvent",
    # Infrastructure
    "ConnectorRegistry",
    # Orchestrator
    "Polaris",
    "PolarisConfig",
    # Sub-modules (for advanced use / testing)
    "ComponentBuilder",
    "AdaptationPipeline",
    "ConfigReloader",
    "MonitoringLoop",
    "MetaLearningLoop",
    "MetricsExportLoop",
]
