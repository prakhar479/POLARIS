"""Polaris abstractions package - core interfaces for extensibility."""

from polaris.abstractions.connector import Connector
from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.meta_learner import (
    AppliedUpdate,
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus,
)
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.abstractions.world_model import PredictionResult, WorldModel

__all__ = [
    "Connector",
    "AdaptationStrategy",
    "AdaptationContext",
    "ParameterSpec",
    "WorldModel",
    "PredictionResult",
    "KnowledgeStore",
    "MetaLearner",
    "ParameterProposal",
    "PerformanceAnalysis",
    "ProposalStatus",
    "AppliedUpdate",
    "Logger",
    "MetricsCollector",
]
