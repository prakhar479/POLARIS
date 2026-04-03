"""Core domain models for Polaris.

Immutable dataclasses representing system state, actions, and results.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Union


def _utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


class HealthStatus(Enum):
    """System health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ExecutionStatus(Enum):
    """Execution result status."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class MetricValue:
    """Metric value with metadata."""

    name: str
    value: Union[int, float, str, bool]
    unit: Optional[str] = None
    timestamp: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", _utc_now())
        if self.tags is None:
            object.__setattr__(self, "tags", {})


@dataclass(frozen=True)
class SystemState:
    """Current state of a managed system."""

    system_id: str
    timestamp: datetime
    metrics: Dict[str, MetricValue]
    health_status: HealthStatus
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize default metadata after dataclass creation."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class AdaptationAction:
    """Adaptation action to execute on a managed system."""

    action_id: str
    action_type: str
    target_system: str
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 0
    timeout_seconds: Optional[int] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize default values after dataclass creation."""
        if not self.action_id:
            object.__setattr__(self, "action_id", str(uuid.uuid4()))
        if self.created_at is None:
            object.__setattr__(self, "created_at", _utc_now())

        # Set default parameters if None
        if self.parameters is None:
            object.__setattr__(self, "parameters", {})

        # Validate required fields
        if not self.action_type:
            raise ValueError("action_type is required")
        if not self.target_system:
            raise ValueError("target_system is required")


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing an adaptation action."""

    action_id: str
    status: ExecutionStatus
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize default completion time after dataclass creation."""
        if self.completed_at is None:
            object.__setattr__(self, "completed_at", _utc_now())
