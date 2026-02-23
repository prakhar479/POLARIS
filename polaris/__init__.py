"""Polaris - Modular Self-Adaptive Systems Framework."""

from typing import Any

__version__ = "2.0.0"

# Abstractions
from polaris.abstractions import (
    AdaptationStrategy,
    Connector,
    KnowledgeStore,
    MetaLearner,
    WorldModel,
)

# Core exports
from polaris.core import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    Polaris,
    SystemState,
)
from polaris.knowledge import InMemoryKnowledgeStore

# Default implementations
from polaris.strategies import ThresholdReactiveStrategy
from polaris.world_model import StatisticalWorldModel

__all__ = [
    # Core
    "Polaris",
    "SystemState",
    "AdaptationAction",
    "ExecutionResult",
    "MetricValue",
    "HealthStatus",
    "ExecutionStatus",
    # Abstractions
    "Connector",
    "AdaptationStrategy",
    "WorldModel",
    "KnowledgeStore",
    "MetaLearner",
    # Default implementations
    "ThresholdReactiveStrategy",
    "StatisticalWorldModel",
    "InMemoryKnowledgeStore",
    "StatisticalMetaLearner",
    "LLMMetaLearner",
]


def __getattr__(name: str) -> Any:
    """Lazy-load optional meta-learners to avoid hard dependency at import time."""
    if name in {"StatisticalMetaLearner", "LLMMetaLearner"}:
        from polaris.meta_learner import LLMMetaLearner, StatisticalMetaLearner

        return {
            "StatisticalMetaLearner": StatisticalMetaLearner,
            "LLMMetaLearner": LLMMetaLearner,
        }[name]
    raise AttributeError(f"module 'polaris' has no attribute {name!r}")
