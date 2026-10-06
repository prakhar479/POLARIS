"""Knowledge Store interface for historical data storage."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple

from polaris.core.models import (
    ActionWorkflow,
    AdaptationAction,
    ExecutionResult,
    SystemState,
)


class KnowledgeStore(ABC):
    """Interface for storing and querying historical data.

    Implement this to customize storage backend.
    """

    @abstractmethod
    async def store_state(self, state: SystemState) -> None:
        """Store system state.

        Args:
            state: System state to store
        """
        pass

    @abstractmethod
    async def store_action(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Store adaptation action and its result.

        Args:
            action: Adaptation action
            result: Execution result
        """
        pass

    @abstractmethod
    async def query_states(
        self, system_id: str, start_time: datetime, end_time: datetime
    ) -> List[SystemState]:
        """Query historical states for a time range.

        Args:
            system_id: System to query
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of system states in time range
        """
        pass

    async def query_actions(
        self, system_id: str, start_time: datetime, end_time: datetime
    ) -> List[Tuple[AdaptationAction, ExecutionResult]]:
        """Query adaptation history.

        Returns:
            List of (action, result) tuples
        """
        return []

    async def store_workflow(self, workflow: ActionWorkflow) -> None:
        """Store action workflow lifecycle data.

        Default implementation is a no-op for backward compatibility.
        """
        return None

    async def query_workflows(self, system_id: str, limit: int = 100) -> List[ActionWorkflow]:
        """Query action workflows for a system.

        Default implementation returns an empty list.
        """
        return []
