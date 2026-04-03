"""Polaris abstractions package - core interfaces for extensibility."""

from polaris.abstractions.connector import Connector
from polaris.abstractions.connector_capabilities import (
    ConnectorCapabilities,
    normalize_action_token,
)
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
from polaris.abstractions.system_contract import SystemContract
from polaris.abstractions.world_model import PredictionResult, WorldModel

__all__ = [
    "Connector",
    "ConnectorCapabilities",
    "SystemContract",
    "normalize_action_token",
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
