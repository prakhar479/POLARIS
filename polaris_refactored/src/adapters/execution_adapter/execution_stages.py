"""
Execution stages for the execution adapter.

This module defines the execution stages used by the execution adapter to
execute adaptation actions. The stages are defined as classes that implement
the `execute` method. Each stage is responsible for a specific aspect of the
execution pipeline, such as validating preconditions, executing the action,
performing post-condition checks, and so on.

The execution stages are executed asynchronously, with each stage being
executed in an `asyncio` task. This allows for concurrent execution of
multiple stages and enables the execution adapter to handle large numbers of
simultaneous execution requests.

For more information on the execution stages and their usage, see the
`ActionExecutionPipeline` class in the `polaris_refactored.adapters.execution_adapter`
module.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Optional, Dict, Any

from ...domain.models import AdaptationAction, ExecutionResult, ExecutionStatus, SystemState
from ...domain.interfaces import ManagedSystemConnector


logger = logging.getLogger(__name__)


class ExecutionStage(ABC):
    """Base class for execution pipeline stages."""
    
    @abstractmethod
    async def execute(self, action: AdaptationAction, context: dict) -> dict:
        """Execute this stage of the pipeline."""
        pass


class ValidationStage(ExecutionStage):
    """Validates the adaptation action and connector preconditions."""
    
    def __init__(self, require_supported_action: bool = True):
        self.require_supported_action = require_supported_action
    
    async def execute(self, action: AdaptationAction, context: dict) -> dict:
        connector: ManagedSystemConnector = context.get("connector")
        if connector is None:
            raise ValueError("Connector not resolved in context for ValidationStage")
        
        # Basic action validation
        if not action.action_type:
            raise ValueError("Action type is required")
        if not action.target_system:
            raise ValueError("Action target_system is required")
        
        # Ensure action is supported
        if self.require_supported_action:
            try:
                supported = await connector.get_supported_actions()
                if action.action_type not in supported:
                    raise ValueError(f"Unsupported action_type '{action.action_type}' for target '{action.target_system}'")
            except Exception as e:
                raise ValueError(f"Failed to verify supported actions: {e}")
        
        # Connector-level validation
        try:
            valid = await connector.validate_action(action)
            if not valid:
                raise ValueError("Connector validation failed for action")
        except Exception as e:
            raise ValueError(f"Connector validate_action error: {e}")
        
        context.setdefault("stage_results", {})[self.__class__.__name__] = "success"
        return context


class PreConditionCheckStage(ExecutionStage):
    """Checks system preconditions before executing action."""
    
    def __init__(self, rules: Optional[Dict[str, Dict[str, Any]]] = None):
        self.rules = rules or {}
    
    async def execute(self, action: AdaptationAction, context: dict) -> dict:
        connector: ManagedSystemConnector = context.get("connector")
        if connector is None:
            raise ValueError("Connector not resolved in context for PreConditionCheckStage")
        
        try:
            current_state: SystemState = await connector.get_system_state()
            context["pre_state"] = current_state
        except Exception as e:
            raise ValueError(f"Failed to get current system state: {e}")
        
        # Apply optional action-specific rules
        action_rules = self.rules.get(action.action_type, {})
        # Example rule hooks could be validated here; keeping generic for now
        # If any rule fails, raise a ValueError with detail
        
        context.setdefault("stage_results", {})[self.__class__.__name__] = "success"
        return context


class ActionExecutionStage(ExecutionStage):
    """Executes the action on the managed system via connector."""
    
    def __init__(self, timeout_seconds: Optional[int] = None):
        self.timeout_seconds = timeout_seconds
    
    async def execute(self, action: AdaptationAction, context: dict) -> dict:
        connector: ManagedSystemConnector = context.get("connector")
        if connector is None:
            raise ValueError("Connector not resolved in context for ActionExecutionStage")
        
        # Use own timeout if set; otherwise fall back to action.timeout_seconds.
        timeout = self.timeout_seconds if self.timeout_seconds is not None else action.timeout_seconds
        try:
            if timeout is not None:
                result: ExecutionResult = await asyncio.wait_for(connector.execute_action(action), timeout=timeout)
            else:
                result = await connector.execute_action(action)
        except asyncio.TimeoutError:
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.TIMEOUT,
                result_data={"reason": "action execution timed out"},
                error_message="Timeout during action execution",
            )
        except Exception as e:
            result = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data={"reason": "action execution error"},
                error_message=str(e),
            )
        
        context["execution_result"] = result
        context.setdefault("stage_results", {})[self.__class__.__name__] = "success" if result.status == ExecutionStatus.SUCCESS else "failed"
        return context


class PostExecutionVerificationStage(ExecutionStage):
    """Verifies post-conditions after action execution."""
    
    def __init__(self, rules: Optional[Dict[str, Dict[str, Any]]] = None):
        self.rules = rules or {}
    
    async def execute(self, action: AdaptationAction, context: dict) -> dict:
        connector: ManagedSystemConnector = context.get("connector")
        if connector is None:
            raise ValueError("Connector not resolved in context for PostExecutionVerificationStage")
        
        # If prior stage already failed, skip verification but mark appropriately
        exec_result: Optional[ExecutionResult] = context.get("execution_result")
        if not exec_result or exec_result.status != ExecutionStatus.SUCCESS:
            context.setdefault("stage_results", {})[self.__class__.__name__] = "skipped"
            return context
        
        # Obtain post state and verify optional rules
        try:
            post_state: SystemState = await connector.get_system_state()
            context["post_state"] = post_state
        except Exception as e:
            # Degrade to partial if verification cannot be performed
            degraded = ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.PARTIAL,
                result_data=exec_result.result_data,
                error_message=f"Post verification failed: {e}",
            )
            context["execution_result"] = degraded
            context.setdefault("stage_results", {})[self.__class__.__name__] = "failed"
            return context
        
        # Example: rule-based checks could be performed here
        context.setdefault("stage_results", {})[self.__class__.__name__] = "success"
        return context