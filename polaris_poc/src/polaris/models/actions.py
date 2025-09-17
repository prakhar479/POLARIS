"""
Control action data models for POLARIS framework.

This module defines the structured models for control actions and their
execution results in the POLARIS adaptation system.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ActionType(str, Enum):
    """
    Enumeration of supported control action types.
    
    These represent the different types of adaptation actions
    that can be executed on managed systems.
    """
    ADD_SERVER = "ADD_SERVER"
    REMOVE_SERVER = "REMOVE_SERVER"
    ADJUST_QOS = "ADJUST_QOS"
    SET_DIMMER = "SET_DIMMER"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    RESTART = "RESTART"
    RECONFIGURE = "RECONFIGURE"
    CUSTOM = "CUSTOM"


class ActionPriority(str, Enum):
    """Priority levels for control actions."""
    LOW = "low"
    NORMAL = "normal"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionStatus(str, Enum):
    """Status of action execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ControlAction(BaseModel):
    """
    Structure for control actions from POLARIS.
    
    This model represents an adaptation action to be executed on a managed system,
    including its type, parameters, and metadata.
    """
    
    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this action"
    )
    
    action_type: str = Field(
        ...,
        description="Type of the action to execute"
    )
    
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp of when the action was created"
    )
    
    source: str = Field(
        default="unknown",
        description="Source component that generated this action"
    )
    
    target: Optional[str] = Field(
        default=None,
        description="Target system or component for this action"
    )
    
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action"
    )
    
    priority: ActionPriority = Field(
        default=ActionPriority.NORMAL,
        description="Priority of this action"
    )
    
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout for action execution in seconds"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the action"
    )

    @field_validator('timestamp',mode="before")
    def set_timestamp(cls, v):
        """Set timestamp to current UTC time if not provided."""
        if v is None:
            return datetime.now(timezone.utc).isoformat()
        return v
    
    @field_validator('action_type')
    def validate_action_type(cls, v):
        """Ensure action type is uppercase."""
        return v.upper()
    
    @classmethod
    def from_json(cls, json_str: str) -> "ControlAction":
        """Create ControlAction from JSON string."""
        import json
        data = json.loads(json_str)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(exclude_none=True)
    
    # Pydantic v2 model config
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "action_id": "123e4567-e89b-12d3-a456-426614174000",
                "action_type": "ADD_SERVER",
                "timestamp": "2025-08-14T10:30:00Z",
                "source": "coordinator",
                "target": "swim-cluster",
                "params": {
                    "server_type": "compute",
                    "count": 2
                },
                "priority": "normal"
            }
        }
    )


class ExecutionResult(BaseModel):
    """
    Result of action execution.
    
    This model captures the outcome of executing a control action,
    including success status, timing information, and any output or errors.
    """
    
    action_id: str = Field(
        ...,
        description="ID of the action that was executed"
    )
    
    action_type: str = Field(
        ...,
        description="Type of the action that was executed"
    )
    
    status: ActionStatus = Field(
        ...,
        description="Status of the execution"
    )
    
    success: Optional[bool] = Field(
        default=None,
        description="Whether the action succeeded"
    )
    
    message: str = Field(
        default="",
        description="Result message or error description"
    )
    
    started_at: str = Field(
        ...,
        description="ISO 8601 timestamp when execution started"
    )
    
    finished_at: str = Field(
        ...,
        description="ISO 8601 timestamp when execution finished"
    )
    
    duration_sec: float = Field(
        ...,
        description="Duration of execution in seconds"
    )
    
    output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output data from the action execution"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts made"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the execution"
    )

    @field_validator('success', mode="before")
    def set_success(cls, v):
        """Keep provided 'success' value if present; otherwise allow model-level validator to set it."""
        if v is not None:
            return v
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)
    
    def to_json(self) -> bytes:
        """Convert to JSON bytes for NATS publishing."""
        # Use pydantic's JSON serializer then encode to bytes for NATS
        return self.model_dump_json(exclude_none=True).encode()
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "action_id": "123e4567-e89b-12d3-a456-426614174000",
                "action_type": "ADD_SERVER",
                "status": "success",
                "success": True,
                "message": "Successfully added 2 servers",
                "started_at": "2025-08-14T10:30:00Z",
                "finished_at": "2025-08-14T10:30:05Z",
                "duration_sec": 5.0,
                "output": {
                    "servers_added": ["server-03", "server-04"],
                    "new_total": 4
                }
            }
        }
    )

    @model_validator(mode='after')
    def _set_success_from_status(self) -> 'ExecutionResult':
        """After model validation, ensure 'success' reflects the 'status' when not provided."""
        # If success is missing or None, derive it from status
        if self.success is None and self.status is not None:
            self.success = (self.status == ActionStatus.SUCCESS)
    
        return self
