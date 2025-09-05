"""
Adapter Layer - Interface between POLARIS and managed systems

This layer provides adapters that interface with external managed systems,
implementing the Template Method pattern for consistent adapter behavior.
"""

from .base_adapter import PolarisAdapter
from .monitor_adapter import MonitorAdapter, MetricCollectionStrategy
from .execution_adapter import ExecutionAdapter, ActionExecutionPipeline

__all__ = [
    "PolarisAdapter",
    "MonitorAdapter",
    "MetricCollectionStrategy",
    "ExecutionAdapter", 
    "ActionExecutionPipeline",
]