"""
Abstract Meta-Learner Agent interface for POLARIS Framework.

This module defines the abstract interface for meta-level learning agents that operate
at the adaptation-strategy level, continuously improving the POLARIS system's adaptation
capabilities through analysis of aggregated insights and reasoning patterns.

The Meta-Learner Agent is designed to:
- Analyze long-term adaptation patterns and outcomes from the knowledge base
- Calibrate and align the world model (digital twin) with observed system behavior
- Update adaptation system parameters (utility weights, policy parameters, coordination strategies)
- Provide confidence estimates for proposed parameter updates
- Support multiple triggering mechanisms (periodic, event-driven, performance-driven)
- Remain implementation-agnostic for different meta-learning approaches
"""

import abc
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TriggerType(str, Enum):
    """Types of triggers that can initiate meta-learning cycles."""

    PERIODIC = "periodic"
    EVENT_DRIVEN = "event_driven"
    PERFORMANCE_DRIVEN = "performance_driven"
    THRESHOLD_VIOLATION = "threshold_violation"
    MANUAL = "manual"


class ParameterType(str, Enum):
    """Types of adaptation parameters that can be updated."""

    UTILITY_WEIGHTS = "utility_weights"
    POLICY_PARAMETERS = "policy_parameters"
    COORDINATION_STRATEGIES = "coordination_strategies"
    THRESHOLD_VALUES = "threshold_values"
    CONTROL_GAINS = "control_gains"
    LEARNING_RATES = "learning_rates"


class MetaLearningContext(BaseModel):
    """Context information for meta-learning operations."""

    trigger_type: TriggerType
    trigger_source: str = Field(
        description="Source component that triggered the learning cycle"
    )
    time_window_hours: float = Field(
        default=24.0, description="Time window for analysis"
    )
    focus_areas: List[str] = Field(
        default_factory=list, description="Specific areas to focus on"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="Constraints for updates"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )


class ParameterUpdate(BaseModel):
    """Represents a proposed parameter update with confidence."""

    update_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parameter_type: ParameterType
    parameter_path: str = Field(
        description="Dot-notation path to the parameter (e.g., 'coordinator.utility.performance')"
    )
    old_value: Any
    new_value: Any
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in this update (0.0 to 1.0)"
    )
    reasoning: str = Field(description="Human-readable explanation for the update")
    expected_impact: str = Field(description="Expected impact on system behavior")
    risk_assessment: str = Field(description="Risk assessment for this change")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationRequest(BaseModel):
    """Request for world model calibration."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_metrics: List[str] = Field(description="Metrics to focus calibration on")
    calibration_data: Dict[str, Any] = Field(
        description="Historical data for calibration"
    )
    validation_window_hours: float = Field(
        default=1.0, description="Time window for validation"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CalibrationResult(BaseModel):
    """Result of world model calibration."""

    request_id: str
    success: bool
    improvement_score: float = Field(
        ge=0.0, le=1.0, description="Improvement in model accuracy"
    )
    calibrated_parameters: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MetaLearningInsights(BaseModel):
    """Aggregated insights from meta-learning analysis."""

    insights_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    analysis_window: Dict[str, datetime] = Field(
        description="Start and end times of analysis"
    )
    adaptation_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    performance_trends: Dict[str, float] = Field(default_factory=dict)
    anomaly_frequencies: Dict[str, int] = Field(default_factory=dict)
    coordination_effectiveness: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    confidence_overall: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in insights"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaseMetaLearnerAgent(ABC):
    """
    Abstract base class for Meta-Learner agents in POLARIS.

    This interface defines the contract for meta-level learning components that
    operate at the adaptation-strategy level, focusing on improving the overall
    adaptation system's effectiveness rather than handling individual adaptation outcomes.

    Key responsibilities:
    - Analyze aggregated adaptation patterns and outcomes
    - Calibrate the world model to maintain alignment with system behavior
    - Update adaptation system parameters based on learned insights
    - Provide confidence estimates for proposed changes
    - Support multiple triggering mechanisms for learning cycles
    """

    def __init__(
        self,
        agent_id: str,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Meta-Learner agent.

        Args:
            agent_id: Unique identifier for this agent instance
            logger: Logger instance (created if not provided)
            config: Agent-specific configuration
        """
        self.agent_id = agent_id
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.running = False

    @abstractmethod
    async def analyze_adaptation_patterns(
        self, context: MetaLearningContext
    ) -> MetaLearningInsights:
        """
        Analyze adaptation patterns and system behavior from the knowledge base.

        This method extracts and analyzes aggregated insights about adaptation
        effectiveness, system performance trends, and recurring patterns.

        Args:
            context: Context information for the analysis

        Returns:
            MetaLearningInsights: Aggregated insights and recommendations

        Raises:
            MetaLearningError: If analysis fails
        """
        pass

    @abstractmethod
    async def calibrate_world_model(
        self, calibration_request: CalibrationRequest
    ) -> CalibrationResult:
        """
        Calibrate the world model (digital twin) to improve its accuracy.

        This method interacts with the world model to align it with observed
        system behavior, ensuring the digital twin remains accurate over time.

        Args:
            calibration_request: Request specifying calibration parameters

        Returns:
            CalibrationResult: Result of the calibration process

        Raises:
            CalibrationError: If calibration fails
        """
        pass

    @abstractmethod
    async def propose_parameter_updates(
        self, insights: MetaLearningInsights, context: MetaLearningContext
    ) -> List[ParameterUpdate]:
        """
        Propose updates to adaptation system parameters based on insights.

        This method generates specific parameter updates (utility weights,
        policy parameters, coordination strategies) with confidence estimates.

        Args:
            insights: Aggregated insights from pattern analysis
            context: Context information for the learning cycle

        Returns:
            List[ParameterUpdate]: List of proposed parameter updates

        Raises:
            ParameterUpdateError: If parameter update generation fails
        """
        pass

    @abstractmethod
    async def validate_updates(
        self,
        proposed_updates: List[ParameterUpdate],
        validation_context: Optional[Dict[str, Any]] = None,
    ) -> List[ParameterUpdate]:
        """
        Validate proposed parameter updates before application.

        This method performs safety checks, constraint validation, and
        impact assessment for proposed parameter changes.

        Args:
            proposed_updates: List of proposed parameter updates
            validation_context: Additional context for validation

        Returns:
            List[ParameterUpdate]: Validated updates (may be filtered or modified)

        Raises:
            ValidationError: If validation fails
        """
        pass

    @abstractmethod
    async def apply_updates(
        self, validated_updates: List[ParameterUpdate]
    ) -> Dict[str, bool]:
        """
        Apply validated parameter updates to the adaptation system.

        This method executes the actual parameter updates and tracks
        their application status.

        Args:
            validated_updates: List of validated parameter updates

        Returns:
            Dict[str, bool]: Mapping of update_id to success status

        Raises:
            UpdateApplicationError: If update application fails
        """
        pass

    @abstractmethod
    async def handle_trigger(
        self, trigger_type: TriggerType, trigger_data: Dict[str, Any]
    ) -> bool:
        """
        Handle different types of meta-learning triggers.

        This method processes various trigger types and initiates appropriate
        meta-learning cycles based on the trigger characteristics.

        Args:
            trigger_type: Type of trigger that occurred
            trigger_data: Data associated with the trigger

        Returns:
            bool: True if trigger was handled successfully

        Raises:
            TriggerHandlingError: If trigger handling fails
        """
        pass

    # Optional methods that can be overridden for specific implementations

    async def get_learning_status(self) -> Dict[str, Any]:
        """
        Get current meta-learning status and statistics.

        Returns:
            Dict[str, Any]: Status information including recent activities,
                           performance metrics, and configuration state
        """
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "last_analysis": None,
            "last_calibration": None,
            "last_update": None,
            "total_updates_applied": 0,
            "average_confidence": 0.0,
        }

    async def configure_triggers(
        self, trigger_config: Dict[TriggerType, Dict[str, Any]]
    ) -> bool:
        """
        Configure trigger mechanisms for meta-learning cycles.

        Args:
            trigger_config: Configuration for different trigger types

        Returns:
            bool: True if configuration was successful
        """
        self.logger.info(f"Configuring triggers: {list(trigger_config.keys())}")
        return True

    async def export_learning_history(
        self, time_window_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Export historical learning data for analysis or backup.

        Args:
            time_window_hours: Time window for export (None for all history)

        Returns:
            Dict[str, Any]: Historical learning data
        """
        return {
            "agent_id": self.agent_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "time_window_hours": time_window_hours,
            "history": [],
        }


# Custom exceptions for meta-learning operations


class MetaLearningError(Exception):
    """Base exception for meta-learning operations."""

    pass


class CalibrationError(MetaLearningError):
    """Raised when world model calibration fails."""

    pass


class ParameterUpdateError(MetaLearningError):
    """Raised when parameter update generation fails."""

    pass


class ValidationError(MetaLearningError):
    """Raised when parameter update validation fails."""

    pass


class UpdateApplicationError(MetaLearningError):
    """Raised when parameter update application fails."""

    pass


class TriggerHandlingError(MetaLearningError):
    """Raised when trigger handling fails."""

    pass
