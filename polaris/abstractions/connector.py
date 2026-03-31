"""Connector interface for integrating managed systems with Polaris."""

from abc import ABC, abstractmethod
from typing import List

from polaris.abstractions.connector_capabilities import ConnectorCapabilities
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState


class Connector(ABC):
    """Interface for integrating managed systems with Polaris.

    Implement this interface to connect your system to the Polaris framework.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the managed system.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the managed system.

        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_system_id(self) -> str:
        """Get unique identifier for this managed system.

        Returns:
            str: Unique system identifier
        """
        pass

    @abstractmethod
    async def collect_telemetry(self) -> SystemState:
        """Collect current system state and metrics.

        Returns:
            SystemState: Current system state with metrics
        """
        pass

    @abstractmethod
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action on the managed system.

        Args:
            action: The adaptation action to execute

        Returns:
            ExecutionResult: Result of the action execution
        """
        pass

    @abstractmethod
    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if an adaptation action can be executed.

        Args:
            action: The adaptation action to validate

        Returns:
            bool: True if action is valid, False otherwise
        """
        pass

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """Get list of action types supported by this managed system.

        Returns:
            List[AdaptationAction]: List of supported action objects
        """
        return []

    async def get_capabilities(self) -> ConnectorCapabilities:
        """Get normalized connector capabilities used by runtime contracts."""
        supported_actions = await self.get_supported_actions()
        action_types = []
        for action in supported_actions or []:
            action_type = getattr(action, "action_type", None)
            if isinstance(action_type, str) and action_type.strip():
                action_types.append(action_type.strip())

        return ConnectorCapabilities.from_supported_action_types(action_types)
