"""System contract model shared across orchestration, strategies, and tools."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from polaris.abstractions.connector_capabilities import ConnectorCapabilities


class MetricType(str, Enum):
    """Metric classification."""

    GAUGE = "gauge"
    COUNTER = "counter"
    RATE = "rate"
    HISTOGRAM = "histogram"


class MetricDirection(str, Enum):
    """Optimization direction for a metric."""

    MINIMIZE = "minimize"  # e.g., latency, error_rate, memory_leak
    MAXIMIZE = "maximize"  # e.g., throughput, cache_hit_rate, battery_level
    STABILIZE = "stabilize"  # e.g., cpu_utilization around 60%, buffer_level


@dataclass(frozen=True)
class ActionSchema:
    """Specification of an adaptation action supported by a connector."""

    action_type: str
    description: str = ""
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    required_parameters: Tuple[str, ...] = ()
    rollback_action: Optional[str] = None
    expected_duration_seconds: float = 0.0
    default_verification_window_seconds: float = 0.0
    performance_impact: str = "neutral"  # "positive", "negative", "neutral"
    cost_impact: str = "neutral"  # "positive" (saves cost), "negative" (increases cost), "neutral"
    qos_impact: str = "neutral"  # "positive", "negative", "neutral"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_parameters(
        self, parameters: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """Validate parameters against required keys and schema bounds."""
        params = parameters or {}
        for req in self.required_parameters:
            if req not in params:
                return False, f"Missing required parameter '{req}' for action '{self.action_type}'"

        for key, spec in self.parameters_schema.items():
            if key in params and isinstance(spec, dict):
                val = params[key]
                expected_type = spec.get("type")
                if expected_type == "integer" and not isinstance(val, int):
                    return False, f"Parameter '{key}' must be an integer"
                elif expected_type == "number" and not isinstance(val, (int, float)):
                    return False, f"Parameter '{key}' must be numeric"
                if "minimum" in spec and isinstance(val, (int, float)):
                    if val < spec["minimum"]:
                        return False, f"Parameter '{key}' must be >= {spec['minimum']}"
                if "maximum" in spec and isinstance(val, (int, float)):
                    if val > spec["maximum"]:
                        return False, f"Parameter '{key}' must be <= {spec['maximum']}"
        return True, None


@dataclass(frozen=True)
class MetricSchema:
    """Specification of a telemetry metric exposed by a connector."""

    name: str
    metric_type: MetricType = MetricType.GAUGE
    direction: MetricDirection = MetricDirection.MINIMIZE
    unit: Optional[str] = None
    description: str = ""
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass(frozen=True)
class SLOContract:
    """Declarative Service Level Objective contract."""

    metric_name: str
    operator: str  # "<", "<=", ">", ">=", "=="
    target_value: float
    window_seconds: int = 60
    priority: float = 1.0  # Weight in multi-objective evaluation
    description: str = ""

    def is_violated(self, current_value: float) -> bool:
        """Check if current metric value violates the SLO."""
        if self.operator == "<":
            return current_value >= self.target_value
        elif self.operator == "<=":
            return current_value > self.target_value
        elif self.operator == ">":
            return current_value <= self.target_value
        elif self.operator == ">=":
            return current_value < self.target_value
        elif self.operator == "==":
            return current_value != self.target_value
        return False


@dataclass(frozen=True)
class SystemContract:
    """Immutable system-level contract used during adaptation decisions."""

    system_id: str
    connector_type: str = ""
    supported_action_types: Tuple[str, ...] = ()
    action_aliases: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, ActionSchema] = field(default_factory=dict)
    metrics: Dict[str, MetricSchema] = field(default_factory=dict)
    slos: Tuple[SLOContract, ...] = ()
    dependencies: Tuple[str, ...] = ()

    @classmethod
    def from_capabilities(
        cls,
        system_id: str,
        connector_type: str,
        capabilities: ConnectorCapabilities,
        metadata: Dict[str, Any] | None = None,
    ) -> "SystemContract":
        """Create a contract from connector capabilities."""
        merged_metadata = dict(capabilities.metadata)
        merged_metadata.update(metadata or {})

        # Inherit typed actions, metrics, and slos from capabilities if present
        actions = getattr(capabilities, "actions", {}) or {}
        metrics = getattr(capabilities, "metrics", {}) or {}
        slos = getattr(capabilities, "slos", ()) or ()
        dependencies = tuple(getattr(capabilities, "dependencies", ()) or ())
        if not dependencies and metadata and "dependencies" in metadata:
            raw_deps = metadata["dependencies"]
            if isinstance(raw_deps, (list, tuple)):
                dependencies = tuple(raw_deps)

        # Fallback populate actions from supported_action_types if actions dict is empty
        if not actions and capabilities.supported_action_types:
            actions = {
                act_type: ActionSchema(action_type=act_type)
                for act_type in capabilities.supported_action_types
            }

        return cls(
            system_id=system_id,
            connector_type=connector_type,
            supported_action_types=tuple(capabilities.supported_action_types),
            action_aliases=dict(capabilities.action_aliases),
            metadata=merged_metadata,
            actions=dict(actions),
            metrics=dict(metrics),
            slos=tuple(slos),
            dependencies=dependencies,
        )

    def supported_actions_list(self) -> list[str]:
        """Return supported actions as a mutable list copy."""
        return list(self.supported_action_types)

    def get_action_schema(self, action_type: str) -> Optional[ActionSchema]:
        """Retrieve ActionSchema resolving canonical names and aliases."""
        norm = (action_type or "").strip().lower()
        if norm in self.actions:
            return self.actions[norm]
        canonical = self.action_aliases.get(norm)
        if canonical and canonical in self.actions:
            return self.actions[canonical]
        # Match case-insensitive
        for key, schema in self.actions.items():
            if key.lower() == norm or (canonical and key.lower() == canonical.lower()):
                return schema
        return None

    def validate_action(self, action: Any) -> Tuple[bool, Optional[str]]:
        """Validate if an action is supported and its parameters conform to schema."""
        action_type = getattr(action, "action_type", None)
        if not action_type or not isinstance(action_type, str):
            return False, "Action must have a non-empty string action_type"

        schema = self.get_action_schema(action_type)
        if not schema and action_type.lower() not in [
            a.lower() for a in self.supported_action_types
        ]:
            canonical = self.action_aliases.get(action_type.lower())
            if not canonical or canonical.lower() not in [
                a.lower() for a in self.supported_action_types
            ]:
                return False, f"Action type '{action_type}' is not supported by contract"

        if schema:
            params = getattr(action, "parameters", {})
            return schema.validate_parameters(params)

        return True, None

    def get_violated_slos(self, state: Any) -> list[SLOContract]:
        """Return list of violated SLOs for the given system state."""
        violated: list[SLOContract] = []
        metrics = getattr(state, "metrics", {}) or {}
        for slo in self.slos:
            if slo.metric_name in metrics:
                mv = metrics[slo.metric_name]
                raw_val = getattr(mv, "value", None)
                if isinstance(raw_val, (int, float)):
                    if slo.is_violated(float(raw_val)):
                        violated.append(slo)
        return violated
