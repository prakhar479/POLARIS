"""
Execution Adapter Module - Execution Adapter Implementation

This module provides the ExecutionAdapter class that implements the execution
adapter for POLARIS, executing adaptation actions on managed systems.
"""

from .execution_adapter import ExecutionAdapter, ActionExecutionPipeline
from .execution_stages import (
    ExecutionStage, ValidationStage, PreConditionCheckStage, 
    ActionExecutionStage, PostExecutionVerificationStage
)

__all__ = [
    # Execution Adapter
    "ExecutionAdapter",
    "ActionExecutionPipeline",
    
    # Execution Stages
    "ExecutionStage",
    "ValidationStage",
    "PreConditionCheckStage",
    "ActionExecutionStage",
    "PostExecutionVerificationStage"
]
