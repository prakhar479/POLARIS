"""
Core Domain Interfaces

Defines the key interfaces that external systems and internal components must implement.
This preserves the existing ManagedSystemConnector interface design.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from .models import SystemState, AdaptationAction, ExecutionResult, MetricValue


class ManagedSystemConnector(ABC):
    """
    Interface that managed systems must implement to connect with POLARIS.
    
    This preserves the existing interface design for backward compatibility
    while providing a clean contract for managed system integration.
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
        Get the unique identifier for this managed system.
        
        Returns:
            str: Unique system identifier
        """
        pass
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """
        Collect current metrics from the managed system.
        
        Returns:
            Dict[str, MetricValue]: Dictionary of metric name to metric value
        """
        pass
    
    @abstractmethod
    async def get_system_state(self) -> SystemState:
        """
        Get the current state of the managed system.
        
        Returns:
            SystemState: Current system state
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
    
    @abstractmethod
    async def get_supported_actions(self) -> List[str]:
        """
        Get the list of action types supported by this managed system.
        
        Returns:
            List[str]: List of supported action type names
        """
        pass


class AdaptationCommand(ABC):
    """
    Interface for adaptation commands using the Command pattern.
    """
    
    @abstractmethod
    async def execute(self) -> ExecutionResult:
        """
        Execute the adaptation command.
        
        Returns:
            ExecutionResult: Result of the command execution
        """
        pass
    
    @abstractmethod
    async def can_execute(self) -> bool:
        """
        Check if the command can be executed in the current context.
        
        Returns:
            bool: True if command can be executed, False otherwise
        """
        pass
    
    @abstractmethod
    def get_action(self) -> AdaptationAction:
        """
        Get the adaptation action associated with this command.
        
        Returns:
            AdaptationAction: The adaptation action
        """
        pass


class EventHandler(ABC):
    """
    Interface for event handlers in the event-driven architecture.
    """
    
    @abstractmethod
    async def handle(self, event: Any) -> None:
        """
        Handle an event.
        
        Args:
            event: The event to handle
        """
        pass
    
    @abstractmethod
    def can_handle(self, event: Any) -> bool:
        """
        Check if this handler can handle the given event.
        
        Args:
            event: The event to check
            
        Returns:
            bool: True if this handler can handle the event
        """
        pass


class ConfigurationSource(ABC):
    """
    Interface for configuration sources in the configuration system.
    """
    
    @abstractmethod
    async def load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration data from this source.
        
        Returns:
            Dict[str, Any]: Configuration data
        """
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        Get the priority of this configuration source.
        Higher numbers have higher priority.
        
        Returns:
            int: Priority value
        """
        pass