"""World Model interface for system behavior modeling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from polaris.core.models import AdaptationAction, SystemState


@dataclass
class PredictionResult:
    """Result of a world model prediction."""

    predicted_metrics: Dict[str, float]
    confidence: float
    reasoning: str = ""
    uncertainty: Dict[str, float] = field(default_factory=dict)


class WorldModel(ABC):
    """Interface for system behavior modeling and prediction.

    Implement this to customize how Polaris understands system behavior.
    """

    @abstractmethod
    async def update(self, state: SystemState) -> None:
        """Update model with new system state.

        Args:
            state: New system state observation
        """
        pass

    @abstractmethod
    async def predict(
        self, action: AdaptationAction, current_state: SystemState
    ) -> PredictionResult:
        """Predict outcome of executing an action.

        Args:
            action: Action to predict outcome for
            current_state: Current system state

        Returns:
            PredictionResult with predicted metrics and confidence
        """
        pass

    @abstractmethod
    async def get_insights(self) -> Dict[str, Any]:
        """Get insights about system behavior.

        Returns:
            Dict with model insights (trends, patterns, etc.)
        """
        pass

    def is_stressed(self, system_id: str) -> bool:
        """Check if the specified system is experiencing behavioral stress or an anomalous regime.

        Override in concrete implementations to provide domain-specific or
        statistical regime detection. Default implementation returns False.
        """
        return False

    def record_pending_action(self, action: AdaptationAction, state: SystemState) -> Optional[Any]:
        """Record an in-flight action to calibrate effect observation on next telemetry update.

        Override in concrete implementations that support closed-loop delta absorption.
        Default implementation is a no-op.
        """
        return None


class DomainSurrogate(ABC):
    """Abstract protocol for pluggable domain physics, queuing, and surrogate models.

    Enables self-adaptive systems to supply analytical, mathematical, or empirical
    priors for counterfactual reasoning before real-world trial-and-error observations.
    """

    @abstractmethod
    def can_handle(self, system_id: str, action_type: str) -> bool:
        """Check if this surrogate applies to the given system and action type."""
        pass

    @abstractmethod
    def predict_deltas(self, action: AdaptationAction, state: SystemState) -> Dict[str, float]:
        """Predict counterfactual metric deltas resulting from the adaptation action."""
        pass
