"""
Adaptation strategy interface for decision-making.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

from polaris.core.models import SystemState, AdaptationAction


@dataclass
class AdaptationContext:
    """Context information for adaptation decisions."""
    system_id: str
    historical_states: list
    world_model_insights: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ParameterSpec:
    """Specification for a tunable parameter."""
    current_value: Any
    type: type
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[list] = None
    description: str = ""
    kind: Optional[str] = None


class AdaptationStrategy(ABC):
    """
    Interface for adaptation decision-making strategies.

    Implement this to create custom adaptation logic.
    """

    @abstractmethod
    async def assess(
        self,
        state: SystemState,
        context: AdaptationContext
    ) -> Optional[AdaptationAction]:
        """
        Assess system state and decide on adaptation.

        Args:
            state: Current system state with metrics
            context: Additional context (history, world model, etc.)

        Returns:
            AdaptationAction if adaptation needed, None otherwise
        """
        pass

    async def on_action_executed(
        self,
        action: AdaptationAction,
        result
    ) -> None:
        """Hook called after action execution (optional)."""
        pass

    # Tuning interface for Meta-Learner

    @abstractmethod
    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """
        Return specification of parameters that can be tuned.

        Returns:
            Dict mapping parameter paths to their specifications
        """
        pass

    @abstractmethod
    async def update_parameter(
        self,
        parameter_path: str,
        new_value: Any
    ) -> bool:
        """
        Update a tunable parameter.

        Args:
            parameter_path: Dot-notation path to parameter
            new_value: New value to set

        Returns:
            True if update succeeded, False otherwise
        """
        pass

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Optional hook to apply configuration updates for hot-reload."""
        return

    async def get_performance_metrics(self) -> Dict[str, float]:
        """
        Optional: Return strategy-specific performance metrics.

        Used by Meta-Learner to assess effectiveness.
        """
        return {}
