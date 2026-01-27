"""
World Model interface for system behavior modeling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass

from polaris.core.models import SystemState, AdaptationAction


@dataclass
class PredictionResult:
    """Result of a world model prediction."""
    predicted_metrics: Dict[str, float]
    confidence: float
    reasoning: str = ""


class WorldModel(ABC):
    """
    Interface for system behavior modeling and prediction.

    Implement this to customize how Polaris understands system behavior.
    """

    @abstractmethod
    async def update(self, state: SystemState) -> None:
        """
        Update model with new system state.

        Args:
            state: New system state observation
        """
        pass

    @abstractmethod
    async def predict(
        self,
        action: AdaptationAction,
        current_state: SystemState
    ) -> PredictionResult:
        """
        Predict outcome of executing an action.

        Args:
            action: Action to predict outcome for
            current_state: Current system state

        Returns:
            PredictionResult with predicted metrics and confidence
        """
        pass

    @abstractmethod
    async def get_insights(self) -> Dict[str, Any]:
        """
        Get insights about system behavior.

        Returns:
            Dict with model insights (trends, patterns, etc.)
        """
        pass
