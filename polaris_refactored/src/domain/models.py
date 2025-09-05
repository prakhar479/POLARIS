"""
Core Domain Models

Defines the core data structures and value objects used throughout POLARIS.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Union
import uuid


class HealthStatus(Enum):
    """System health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ExecutionStatus(Enum):
    """Execution result status enumeration."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class MetricValue:
    """Represents a metric value with metadata."""
    name: str
    value: Union[int, float, str, bool]
    unit: Optional[str] = None
    timestamp: Optional[datetime] = None
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.utcnow())
        if self.tags is None:
            object.__setattr__(self, 'tags', {})


@dataclass(frozen=True)
class SystemState:
    """Represents the current state of a managed system."""
    system_id: str
    timestamp: datetime
    metrics: Dict[str, MetricValue]
    health_status: HealthStatus
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True)
class AdaptationAction:
    """Represents an adaptation action to be executed on a managed system."""
    action_id: str
    action_type: str
    target_system: str
    parameters: Dict[str, Any]
    priority: int = 0
    timeout_seconds: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.action_id == "":
            object.__setattr__(self, 'action_id', str(uuid.uuid4()))
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())


@dataclass(frozen=True)
class ExecutionResult:
    """Represents the result of executing an adaptation action."""
    action_id: str
    status: ExecutionStatus
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.completed_at is None:
            object.__setattr__(self, 'completed_at', datetime.utcnow())


@dataclass(frozen=True)
class SystemDependency:
    """Represents a dependency relationship between systems."""
    source_system: str
    target_system: str
    dependency_type: str
    strength: float  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Dependency strength must be between 0.0 and 1.0")


@dataclass(frozen=True)
class LearnedPattern:
    """Represents a learned pattern from system behavior."""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    learned_at: datetime
    usage_count: int = 0
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")