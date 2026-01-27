"""Polaris - Modular Self-Adaptive Systems Framework."""

__version__ = "2.0.0"

# Core exports
from polaris.core import (
    Polaris,
    SystemState,
    AdaptationAction,
    ExecutionResult,
    MetricValue,
    HealthStatus,
    ExecutionStatus
)

# Abstractions
from polaris.abstractions import (
    Connector,
    AdaptationStrategy,
    WorldModel,
    KnowledgeStore,
    MetaLearner
)

# Default implementations
from polaris.strategies import ThresholdReactiveStrategy
from polaris.world_model import StatisticalWorldModel
from polaris.knowledge import InMemoryKnowledgeStore
from polaris.meta_learner import StatisticalMetaLearner, LLMMetaLearner

__all__ = [
    # Core
    'Polaris',
    'SystemState',
    'AdaptationAction',
    'ExecutionResult',
    'MetricValue',
    'HealthStatus',
    'ExecutionStatus',

    # Abstractions
    'Connector',
    'AdaptationStrategy',
    'WorldModel',
    'KnowledgeStore',
    'MetaLearner',

    # Default implementations
    'ThresholdReactiveStrategy',
    'StatisticalWorldModel',
    'InMemoryKnowledgeStore',
    'StatisticalMetaLearner',
    'LLMMetaLearner'
]
