"""
Control & Reasoning Layer - MAPE-K loop implementation

This layer contains the core decision-making components including the adaptive controller
and reasoning engine that implement the MAPE-K (Monitor, Analyze, Plan, Execute, Knowledge) loop.
Includes PID-based reactive control strategies for fast mathematical adaptation responses.
"""

from .adaptive_controller import (
    PolarisAdaptiveController, 
    ControlStrategy,
    ReactiveControlStrategy,
    PredictiveControlStrategy,
    LearningControlStrategy,
    AdaptationNeed
)
from .reasoning_engine import PolarisReasoningEngine, ReasoningStrategy
from .pid_controller import PIDController, PIDConfig, MetricHistoryManager
from .pid_reactive_strategy import PIDReactiveStrategy, PIDReactiveConfig
from .pid_strategy_factory import PIDStrategyFactory, create_pid_strategy_from_system_type

__all__ = [
    "PolarisAdaptiveController",
    "ControlStrategy",
    "ReactiveControlStrategy",
    "PredictiveControlStrategy",
    "LearningControlStrategy",
    "AdaptationNeed",
    "PolarisReasoningEngine",
    "ReasoningStrategy",
    "PIDController",
    "PIDConfig",
    "MetricHistoryManager",
    "PIDReactiveStrategy",
    "PIDReactiveConfig",
    "PIDStrategyFactory",
    "create_pid_strategy_from_system_type"
]