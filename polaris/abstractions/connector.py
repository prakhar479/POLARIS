"""
Connector interface for integrating managed systems with Polaris.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from polaris.core.models import SystemState, AdaptationAction, ExecutionResult, MetricValue


class Connector(ABC):
    """
    Interface for integrating managed systems with Polaris.

    Implement this interface to connect your system to the Polaris framework.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the managed system.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the managed system.

        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_system_id(self) -> str:
        """
        Get unique identifier for this managed system.

        Returns:
            str: Unique system identifier
        """
        pass

    @abstractmethod
    async def collect_telemetry(self) -> SystemState:
        """
        Collect current system state and metrics.

        Returns:
            SystemState: Current system state with metrics
        """
        pass

    @abstractmethod
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """
        Execute an adaptation action on the managed system.

        Args:
            action: The adaptation action to execute

        Returns:
            ExecutionResult: Result of the action execution
        """
        pass

    @abstractmethod
    async def validate_action(self, action: AdaptationAction) -> bool:
        """
        Validate if an adaptation action can be executed.

        Args:
            action: The adaptation action to validate

        Returns:
            bool: True if action is valid, False otherwise
        """
        pass

    async def get_supported_actions(self) -> List[AdaptationAction]:
        """
        Get list of action types supported by this managed system.

        Returns:
            List[AdaptationAction]: List of supported action objects
        """
        return []
