"""
Monitor Adapter Module - Monitoring Adapter Implementation

This module provides the MonitorAdapter class that implements the monitoring
adapter for POLARIS, interfacing with managed systems and collecting metrics.
"""

from .monitor_adapter import MonitorAdapter
from .monitor_types import MonitoringTarget, MetricCollectionMode, CollectionResult
from .monitor_strategy import (
    MetricCollectionStrategy, PollingStrategy,
    BatchCollectionStrategy, RetryingStrategyDecorator,
    DirectConnectorStrategy
)

__all__ = [
    # Adapter
    "MonitorAdapter",
    # Types
    "MonitoringTarget",
    "MetricCollectionMode",
    "CollectionResult",
    # Strategies
    "MetricCollectionStrategy",
    "PollingStrategy",
    "BatchCollectionStrategy",
    "DirectConnectorStrategy",
    "RetryingStrategyDecorator"
]
