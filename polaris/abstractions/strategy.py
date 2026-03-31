"""Adaptation strategy interface for decision-making."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.abstractions.system_contract import SystemContract
    from polaris.core.models import AdaptationAction, ExecutionResult, SystemState
else:
    # Use Any as fallback for runtime type checks if models can't be imported
    # (avoiding circular imports)
    AdaptationAction = Any
    ExecutionResult = Any
    SystemState = Any


@dataclass
class AdaptationContext:
    """Context information for adaptation decisions."""

    system_id: str
    historical_states: List["SystemState"]
    world_model_insights: Optional[Dict[str, Any]] = None
    system_contract: Optional["SystemContract"] = None
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
    """Interface for adaptation decision-making strategies.

    Implement this to create custom adaptation logic.
    """

    @abstractmethod
    async def assess(
        self, state: "SystemState", context: AdaptationContext
    ) -> List["AdaptationAction"]:
        """Assess system state and decide on adaptation.

        Args:
            state: Current system state with metrics
            context: Additional context (history, world model, etc.)

        Returns:
            List of AdaptationActions if adaptation needed, empty list otherwise
        """
        pass

    @abstractmethod
    async def on_action_executed(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Call after action execution (optional)."""
        pass

    # Tuning interface for Meta-Learner

    @abstractmethod
    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Return specification of parameters that can be tuned.

        Returns:
            Dict mapping parameter paths to their specifications
        """
        pass

    @abstractmethod
    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter.

        Args:
            parameter_path: Dot-notation path to parameter
            new_value: New value to set

        Returns:
            True if update succeeded, False otherwise
        """
        pass

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates for hot-reload."""
        return

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy-specific performance metrics.

        Used by Meta-Learner to assess effectiveness.
        """
        return {}
