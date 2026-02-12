"""Polaris - Modular Self-Adaptive Systems Framework."""

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
from polaris.meta_learner import LLMMetaLearner, StatisticalMetaLearner

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
