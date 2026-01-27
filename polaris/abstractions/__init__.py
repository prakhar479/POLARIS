"""Polaris abstractions package - core interfaces for extensibility."""

from polaris.abstractions.connector import Connector
from polaris.abstractions.strategy import AdaptationStrategy, AdaptationContext, ParameterSpec
from polaris.abstractions.world_model import WorldModel, PredictionResult
from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.meta_learner import (
    MetaLearner,
    ParameterProposal,
    PerformanceAnalysis,
    ProposalStatus,
    AppliedUpdate
)
from polaris.abstractions.observability import Logger, MetricsCollector

__all__ = [
    'Connector',
    'AdaptationStrategy',
    'AdaptationContext',
    'ParameterSpec',
    'WorldModel',
    'PredictionResult',
    'KnowledgeStore',
    'MetaLearner',
    'ParameterProposal',
    'PerformanceAnalysis',
    'ProposalStatus',
    'AppliedUpdate',
    'Logger',
    'MetricsCollector'
]
