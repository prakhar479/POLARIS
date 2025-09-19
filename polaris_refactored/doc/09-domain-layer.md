# Domain Layer

## Overview

The Domain Layer contains the core business models, interfaces, and domain logic that define the fundamental concepts and behaviors of the POLARIS framework. This layer is independent of technical implementation details and focuses on the essential business rules and entities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Domain Layer                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │     Models      │ │   Interfaces    │ │   Value Objects │   │
│  │ - SystemState   │ │ - Connectors    │ │ - MetricValue   │   │
│  │ - Actions       │ │ - Commands      │ │ - Health Status │   │
│  │ - Results       │ │ - Handlers      │ │ - Dependencies  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Enumerations  │ │     Events      │ │   Exceptions    │   │
│  │ - Status Types  │ │ - Domain Events │ │ - Domain Errors │   │
│  │ - Action Types  │ │ - Event Base    │ │ - Validation    │   │
│  │ - Strategies    │ │ - Handlers      │ │ - Business Rules│   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Domain Models

### SystemState

**Purpose**: Represents the current state of a managed system including metrics, health, and metadata.

```python
@dataclass(frozen=True)
class SystemState:
    """Immutable representation of a system's current state."""
    
    system_id: str
    timestamp: datetime
    metrics: Dict[str, MetricValue]
    health_status: HealthStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate system state after initialization."""
        if not self.system_id:
            raise ValueError("System ID cannot be empty")
        
        if not self.metrics:
            raise ValueError("System state must have at least one metric")
        
        # Validate timestamp is not in the future
        if self.timestamp > datetime.utcnow():
            raise ValueError("System state timestamp cannot be in the future")
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get the numeric value of a specific metric."""
        metric = self.metrics.get(metric_name)
        return metric.value if metric else None
    
    def has_metric(self, metric_name: str) -> bool:
        """Check if system state contains a specific metric."""
        return metric_name in self.metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system state to dictionary representation."""
        return {
            "system_id": self.system_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "health_status": self.health_status.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """Create system state from dictionary representation."""
        return cls(
            system_id=data["system_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics={k: MetricValue.from_dict(v) for k, v in data["metrics"].items()},
            health_status=HealthStatus(data["health_status"]),
            metadata=data.get("metadata", {})
        )
```

### AdaptationAction

**Purpose**: Represents an action that can be performed to adapt a system.

```python
@dataclass(frozen=True)
class AdaptationAction:
    """Immutable representation of an adaptation action."""
    
    action_id: str
    system_id: str
    action_type: str
    parameters: Dict[str, Any]
    priority: ActionPriority
    estimated_impact: Optional[Dict[str, float]] = None
    prerequisites: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    
    def __post_init__(self):
        """Validate adaptation action after initialization."""
        if not self.action_id:
            raise ValueError("Action ID cannot be empty")
        
        if not self.system_id:
            raise ValueError("System ID cannot be empty")
        
        if not self.action_type:
            raise ValueError("Action type cannot be empty")
        
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
    
    def has_parameter(self, param_name: str) -> bool:
        """Check if action has a specific parameter."""
        return param_name in self.parameters
    
    def get_parameter(self, param_name: str, default: Any = None) -> Any:
        """Get action parameter value."""
        return self.parameters.get(param_name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert adaptation action to dictionary representation."""
        return {
            "action_id": self.action_id,
            "system_id": self.system_id,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "estimated_impact": self.estimated_impact,
            "prerequisites": self.prerequisites,
            "timeout_seconds": self.timeout_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptationAction':
        """Create adaptation action from dictionary representation."""
        return cls(
            action_id=data["action_id"],
            system_id=data["system_id"],
            action_type=data["action_type"],
            parameters=data["parameters"],
            priority=ActionPriority(data["priority"]),
            estimated_impact=data.get("estimated_impact"),
            prerequisites=data.get("prerequisites", []),
            timeout_seconds=data.get("timeout_seconds", 300)
        )
```

### ExecutionResult

**Purpose**: Represents the result of executing an adaptation action.

```python
@dataclass(frozen=True)
class ExecutionResult:
    """Immutable representation of action execution result."""
    
    action_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: datetime
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate execution result after initialization."""
        if not self.action_id:
            raise ValueError("Action ID cannot be empty")
        
        if self.end_time < self.start_time:
            raise ValueError("End time cannot be before start time")
    
    @property
    def duration(self) -> timedelta:
        """Get execution duration."""
        return self.end_time - self.start_time
    
    @property
    def duration_seconds(self) -> float:
        """Get execution duration in seconds."""
        return self.duration.total_seconds()
    
    @property
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status in [ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary representation."""
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "result_data": self.result_data,
            "error_message": self.error_message
        }
```

## Value Objects

### MetricValue

**Purpose**: Represents a single metric measurement with metadata.

```python
@dataclass(frozen=True)
class MetricValue:
    """Immutable metric value with metadata."""
    
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metric value after initialization."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")
        
        if not isinstance(self.value, (int, float)):
            raise ValueError("Metric value must be numeric")
        
        if math.isnan(self.value) or math.isinf(self.value):
            raise ValueError("Metric value cannot be NaN or infinite")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric value to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricValue':
        """Create metric value from dictionary representation."""
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tags=data.get("tags", {})
        )
```

### SystemDependency

**Purpose**: Represents a dependency relationship between systems.

```python
@dataclass(frozen=True)
class SystemDependency:
    """Immutable representation of system dependency."""
    
    source_system_id: str
    target_system_id: str
    dependency_type: DependencyType
    strength: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate system dependency after initialization."""
        if not self.source_system_id:
            raise ValueError("Source system ID cannot be empty")
        
        if not self.target_system_id:
            raise ValueError("Target system ID cannot be empty")
        
        if self.source_system_id == self.target_system_id:
            raise ValueError("System cannot depend on itself")
        
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Dependency strength must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system dependency to dictionary representation."""
        return {
            "source_system_id": self.source_system_id,
            "target_system_id": self.target_system_id,
            "dependency_type": self.dependency_type.value,
            "strength": self.strength,
            "metadata": self.metadata
        }
```

## Enumerations

### Health Status

```python
class HealthStatus(Enum):
    """System health status enumeration."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    
    def is_healthy(self) -> bool:
        """Check if status indicates healthy system."""
        return self == HealthStatus.HEALTHY
    
    def is_degraded(self) -> bool:
        """Check if status indicates degraded system."""
        return self in [HealthStatus.WARNING, HealthStatus.CRITICAL]
    
    def severity_level(self) -> int:
        """Get numeric severity level (0=healthy, 3=critical)."""
        severity_map = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.CRITICAL: 2,
            HealthStatus.UNKNOWN: 3
        }
        return severity_map[self]
```

### Execution Status

```python
class ExecutionStatus(Enum):
    """Action execution status enumeration."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    
    def is_terminal(self) -> bool:
        """Check if status is terminal (execution completed)."""
        return self in [
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED
        ]
    
    def is_successful(self) -> bool:
        """Check if status indicates successful execution."""
        return self == ExecutionStatus.SUCCESS
```

### Action Priority

```python
class ActionPriority(Enum):
    """Action priority enumeration."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    
    def numeric_value(self) -> int:
        """Get numeric priority value for sorting."""
        priority_map = {
            ActionPriority.LOW: 1,
            ActionPriority.NORMAL: 2,
            ActionPriority.HIGH: 3,
            ActionPriority.CRITICAL: 4
        }
        return priority_map[self]
```

### Dependency Type

```python
class DependencyType(Enum):
    """System dependency type enumeration."""
    
    FUNCTIONAL = "functional"  # System A needs System B to function
    PERFORMANCE = "performance"  # System A's performance affects System B
    DATA = "data"  # System A provides data to System B
    NETWORK = "network"  # Systems share network resources
    RESOURCE = "resource"  # Systems share compute/storage resources
```

## Domain Interfaces

### ManagedSystemConnector

**Purpose**: Defines the contract for integrating with external managed systems.

```python
class ManagedSystemConnector(ABC):
    """Abstract interface for managed system connectors."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the managed system."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the managed system."""
        pass
    
    @abstractmethod
    async def get_system_id(self) -> str:
        """Get the unique identifier for this system."""
        pass
    
    @abstractmethod
    async def collect_metrics(self) -> Dict[str, MetricValue]:
        """Collect current metrics from the system."""
        pass
    
    @abstractmethod
    async def get_system_state(self) -> SystemState:
        """Get the current complete system state."""
        pass
    
    @abstractmethod
    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """Execute an adaptation action on the system."""
        pass
    
    @abstractmethod
    async def validate_action(self, action: AdaptationAction) -> bool:
        """Validate if an action can be executed."""
        pass
    
    @abstractmethod
    async def get_supported_actions(self) -> List[str]:
        """Get list of supported action types."""
        pass
    
    async def get_health_status(self) -> HealthStatus:
        """Get current health status (default implementation)."""
        try:
            state = await self.get_system_state()
            return state.health_status
        except Exception:
            return HealthStatus.UNKNOWN
```

### AdaptationCommand

**Purpose**: Defines the interface for adaptation commands.

```python
class AdaptationCommand(ABC):
    """Abstract interface for adaptation commands."""
    
    @abstractmethod
    async def execute(self, context: AdaptationContext) -> ExecutionResult:
        """Execute the adaptation command."""
        pass
    
    @abstractmethod
    async def validate(self, context: AdaptationContext) -> ValidationResult:
        """Validate the command can be executed."""
        pass
    
    @abstractmethod
    def get_estimated_impact(self) -> Dict[str, float]:
        """Get estimated impact of executing this command."""
        pass
    
    @abstractmethod
    def get_prerequisites(self) -> List[str]:
        """Get list of prerequisites for this command."""
        pass
```

### EventHandler

**Purpose**: Defines the interface for handling domain events.

```python
class EventHandler(ABC, Generic[T]):
    """Abstract interface for event handlers."""
    
    @abstractmethod
    async def handle(self, event: T) -> None:
        """Handle a domain event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: Type) -> bool:
        """Check if this handler can process the given event type."""
        pass
```

## Domain Events

### SystemStateChangedEvent

```python
@dataclass(frozen=True)
class SystemStateChangedEvent:
    """Event fired when system state changes significantly."""
    
    system_id: str
    previous_state: Optional[SystemState]
    current_state: SystemState
    change_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_metric_changes(self) -> Dict[str, float]:
        """Get changes in metric values."""
        if not self.previous_state:
            return {}
        
        changes = {}
        for metric_name, current_metric in self.current_state.metrics.items():
            if metric_name in self.previous_state.metrics:
                previous_value = self.previous_state.metrics[metric_name].value
                change = current_metric.value - previous_value
                if abs(change) > 0.001:  # Avoid floating point noise
                    changes[metric_name] = change
        
        return changes
```

### AdaptationRequestedEvent

```python
@dataclass(frozen=True)
class AdaptationRequestedEvent:
    """Event fired when adaptation is requested."""
    
    system_id: str
    reason: str
    urgency: ActionPriority
    suggested_actions: List[AdaptationAction]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### AdaptationCompletedEvent

```python
@dataclass(frozen=True)
class AdaptationCompletedEvent:
    """Event fired when adaptation is completed."""
    
    system_id: str
    action: AdaptationAction
    result: ExecutionResult
    impact_assessment: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

## Domain Exceptions

### DomainException

```python
class DomainException(Exception):
    """Base exception for domain-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.utcnow()
```

### SystemNotFoundException

```python
class SystemNotFoundException(DomainException):
    """Exception raised when a system is not found."""
    
    def __init__(self, system_id: str):
        super().__init__(f"System not found: {system_id}")
        self.system_id = system_id
```

### InvalidActionException

```python
class InvalidActionException(DomainException):
    """Exception raised when an action is invalid."""
    
    def __init__(self, action: AdaptationAction, reason: str):
        super().__init__(f"Invalid action {action.action_id}: {reason}")
        self.action = action
        self.reason = reason
```

### ActionExecutionException

```python
class ActionExecutionException(DomainException):
    """Exception raised when action execution fails."""
    
    def __init__(self, action_id: str, error: str):
        super().__init__(f"Action execution failed {action_id}: {error}")
        self.action_id = action_id
        self.error = error
```

## Domain Services

### SystemStateValidator

```python
class SystemStateValidator:
    """Validates system state consistency and business rules."""
    
    @staticmethod
    def validate_state(state: SystemState) -> ValidationResult:
        """Validate system state against business rules."""
        errors = []
        
        # Check required metrics
        required_metrics = ["cpu_usage", "memory_usage"]
        for metric in required_metrics:
            if not state.has_metric(metric):
                errors.append(f"Missing required metric: {metric}")
        
        # Check metric value ranges
        for metric_name, metric in state.metrics.items():
            if metric_name.endswith("_usage") and not 0.0 <= metric.value <= 1.0:
                errors.append(f"Usage metric {metric_name} must be between 0.0 and 1.0")
        
        # Check timestamp validity
        max_age = timedelta(minutes=5)
        if datetime.utcnow() - state.timestamp > max_age:
            errors.append("System state is too old")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
```

### ActionValidator

```python
class ActionValidator:
    """Validates adaptation actions against business rules."""
    
    @staticmethod
    def validate_action(action: AdaptationAction, 
                       system_state: SystemState) -> ValidationResult:
        """Validate adaptation action."""
        errors = []
        
        # Check action type is supported
        supported_actions = [
            "scale_out", "scale_in", "restart", 
            "tune_parameters", "migrate", "failover"
        ]
        if action.action_type not in supported_actions:
            errors.append(f"Unsupported action type: {action.action_type}")
        
        # Validate action-specific parameters
        if action.action_type == "scale_out":
            if not action.has_parameter("instances"):
                errors.append("Scale out action requires 'instances' parameter")
            elif action.get_parameter("instances", 0) <= 0:
                errors.append("Scale out instances must be positive")
        
        # Check system health allows action
        if system_state.health_status == HealthStatus.CRITICAL:
            if action.action_type not in ["restart", "failover"]:
                errors.append("Only restart or failover allowed for critical systems")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
```

## Domain Aggregates

### SystemAggregate

```python
class SystemAggregate:
    """Aggregate root for system-related domain objects."""
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self._current_state: Optional[SystemState] = None
        self._state_history: List[SystemState] = []
        self._pending_actions: List[AdaptationAction] = []
        self._dependencies: List[SystemDependency] = []
        self._domain_events: List[Any] = []
    
    def update_state(self, new_state: SystemState) -> None:
        """Update system state and track changes."""
        if new_state.system_id != self.system_id:
            raise ValueError("State system ID does not match aggregate")
        
        previous_state = self._current_state
        self._current_state = new_state
        self._state_history.append(new_state)
        
        # Raise domain event for significant changes
        if self._is_significant_change(previous_state, new_state):
            event = SystemStateChangedEvent(
                system_id=self.system_id,
                previous_state=previous_state,
                current_state=new_state,
                change_type="significant_change"
            )
            self._domain_events.append(event)
    
    def request_adaptation(self, reason: str, urgency: ActionPriority,
                          suggested_actions: List[AdaptationAction]) -> None:
        """Request system adaptation."""
        event = AdaptationRequestedEvent(
            system_id=self.system_id,
            reason=reason,
            urgency=urgency,
            suggested_actions=suggested_actions,
            context={"current_state": self._current_state.to_dict() if self._current_state else {}}
        )
        self._domain_events.append(event)
    
    def get_domain_events(self) -> List[Any]:
        """Get and clear domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def _is_significant_change(self, previous: Optional[SystemState], 
                              current: SystemState) -> bool:
        """Determine if state change is significant."""
        if not previous:
            return True
        
        # Check health status change
        if previous.health_status != current.health_status:
            return True
        
        # Check significant metric changes (>10% change)
        for metric_name, current_metric in current.metrics.items():
            if metric_name in previous.metrics:
                previous_value = previous.metrics[metric_name].value
                if previous_value > 0:
                    change_percent = abs(current_metric.value - previous_value) / previous_value
                    if change_percent > 0.1:  # 10% change threshold
                        return True
        
        return False
```

---

*Continue to [Adapter Layer](./10-adapter-layer.md) →*