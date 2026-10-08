"""AdaMLS (Adaptive Machine Learning System) baseline strategy.

Implements the QoS-aware model switching baseline from:
Kulkarni et al., "Towards Self-Adaptive Machine Learning-Enabled Systems
Through QoS-Aware Model Switching", IEEE/ACM ASE 2023.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult, ExecutionStatus, SystemState
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector

DEFAULT_MODEL_HIERARCHY: List[str] = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
]


class AdaMLSStrategy(AdaptationStrategy):
    """QoS-Aware Model Switching Strategy (AdaMLS baseline).

    Monitors system latency, CPU utilization, and prediction confidence,
    dynamically switching models along a sorted model hierarchy to maintain
    QoS constraints while maximizing accuracy.
    """

    hybrid_cooldown_exempt: bool = True

    def __init__(
        self,
        model_hierarchy: Optional[Sequence[str]] = None,
        latency_sla: float = 0.15,
        cpu_sla: float = 70.0,
        confidence_target: float = 0.65,
        headroom_factor: float = 0.70,
        cooldown_seconds: float = 5.0,
        action_name: str = "switch_model",
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize AdaMLS strategy.

        Args:
            model_hierarchy: Ordered sequence of models from fastest/least-accurate
                to slowest/most-accurate.
            latency_sla: Latency upper bound in seconds (e.g. 0.15s = 150ms).
            cpu_sla: CPU utilization upper bound in percent (0-100).
            confidence_target: Desired minimum confidence/accuracy (0.0-1.0).
            headroom_factor: Ratio of SLA budget that must be free before upgrading.
            cooldown_seconds: Minimum time between consecutive switches.
            action_name: Action type emitted for model switching.
            logger: Optional structured logger.
            metrics: Optional metrics collector.
        """
        self.model_hierarchy: List[str] = list(model_hierarchy or DEFAULT_MODEL_HIERARCHY)
        if len(self.model_hierarchy) < 2:
            raise ValueError("model_hierarchy must contain at least 2 models")

        self.latency_sla = float(latency_sla)
        self.cpu_sla = float(cpu_sla)
        self.confidence_target = float(confidence_target)
        self.headroom_factor = float(headroom_factor)
        self.cooldown_seconds = float(cooldown_seconds)
        self.action_name = action_name
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()

        self._current_model: Optional[str] = None
        self._last_switch_time: Dict[str, datetime] = {}
        self._total_assessments: int = 0
        self._total_switches: int = 0
        self._sla_violations: int = 0

        if self.logger:
            self.logger.info(
                "AdaMLS strategy initialized",
                models=self.model_hierarchy,
                latency_sla=self.latency_sla,
                confidence_target=self.confidence_target,
            )

    def _get_model_index(self, model_name: str) -> int:
        """Get index of model in hierarchy, falling back to median index."""
        try:
            return self.model_hierarchy.index(model_name)
        except ValueError:
            return len(self.model_hierarchy) // 2

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess QoS metrics and decide whether to switch models."""
        self._total_assessments += 1
        now = datetime.now(timezone.utc)
        system_id = state.system_id

        # Determine currently active model
        active_model = state.metadata.get("active_model") if state.metadata else None
        if isinstance(active_model, str) and active_model in self.model_hierarchy:
            self._current_model = active_model
        elif self._current_model is None:
            # Default to median model
            self._current_model = self.model_hierarchy[len(self.model_hierarchy) // 2]

        current_idx = self._get_model_index(self._current_model)

        # Enforce cooldown per system
        last_switch = self._last_switch_time.get(system_id)
        if last_switch is not None:
            elapsed = (now - last_switch).total_seconds()
            if elapsed < self.cooldown_seconds:
                if self.logger:
                    self.logger.debug(
                        "AdaMLS in cooldown",
                        system_id=system_id,
                        elapsed=elapsed,
                        cooldown=self.cooldown_seconds,
                    )
                return []

        # Extract QoS metrics defensively
        def _get_metric_float(*keys: str) -> Optional[float]:
            for k in keys:
                mv = state.metrics.get(k)
                if mv is not None and mv.value is not None:
                    try:
                        return float(mv.value)
                    except (ValueError, TypeError):
                        pass
            return None

        latency = _get_metric_float("response_time", "latency")
        cpu = _get_metric_float("cpu_usage", "cpu")
        confidence = _get_metric_float("confidence_mean", "confidence", "accuracy")

        target_idx: Optional[int] = None
        reasoning = ""

        # Condition 1: Latency or CPU violation -> Downgrade model (step down)
        latency_violation = latency is not None and latency > self.latency_sla
        cpu_violation = cpu is not None and cpu > self.cpu_sla

        if latency_violation or cpu_violation:
            self._sla_violations += 1
            if current_idx > 0:
                # If severe violation, drop 2 steps, else 1
                step = (
                    2 if (latency and latency > self.latency_sla * 1.3 and current_idx >= 2) else 1
                )
                target_idx = max(0, current_idx - step)
                reasoning = (
                    f"AdaMLS downgrade: latency={latency}s (SLA={self.latency_sla}s), "
                    f"cpu={cpu}% (SLA={self.cpu_sla}%)"
                )

        # Condition 2: Headroom available & confidence below target -> Upgrade model (step up)
        elif (
            latency is not None
            and latency < self.latency_sla * self.headroom_factor
            and (cpu is None or cpu < self.cpu_sla * self.headroom_factor)
            and confidence is not None
            and confidence < self.confidence_target
        ):
            if current_idx < len(self.model_hierarchy) - 1:
                target_idx = current_idx + 1
                reasoning = (
                    f"AdaMLS upgrade: confidence={confidence:.3f} < target {self.confidence_target:.3f} "
                    f"with latency headroom ({latency:.3f}s < {self.latency_sla * self.headroom_factor:.3f}s)"
                )

        if target_idx is not None and target_idx != current_idx:
            target_model = self.model_hierarchy[target_idx]
            action = AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type=self.action_name,
                target_system=system_id,
                parameters={"model_name": target_model},
                priority=1,
                metadata={"reasoning": reasoning},
            )
            self._last_switch_time[system_id] = now
            self._total_switches += 1
            self.metrics.increment(
                "polaris.strategy.adamls.switch_proposed",
                tags={"from": self._current_model, "to": target_model},
            )
            return [action]

        return []

    async def on_action_executed(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Handle notification of executed adaptation action."""
        if result.status == ExecutionStatus.SUCCESS:
            new_model = (action.parameters or {}).get("model_name")
            if isinstance(new_model, str) and new_model in self.model_hierarchy:
                self._current_model = new_model
                if self.logger:
                    self.logger.info(
                        "AdaMLS updated active model after successful switch",
                        active_model=new_model,
                    )

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Expose tunable parameters for meta-learning optimization."""
        return {
            "latency_sla": ParameterSpec(
                current_value=self.latency_sla,
                type=float,
                min_value=0.01,
                max_value=1.0,
                description="Latency SLA upper bound in seconds",
                kind="threshold_high",
            ),
            "cpu_sla": ParameterSpec(
                current_value=self.cpu_sla,
                type=float,
                min_value=10.0,
                max_value=95.0,
                description="CPU usage upper bound percentage",
                kind="threshold_high",
            ),
            "confidence_target": ParameterSpec(
                current_value=self.confidence_target,
                type=float,
                min_value=0.1,
                max_value=0.99,
                description="Target detection confidence",
                kind="threshold_low",
            ),
            "cooldown_seconds": ParameterSpec(
                current_value=self.cooldown_seconds,
                type=float,
                min_value=1.0,
                max_value=60.0,
                description="Minimum seconds between model switches",
                kind="cooldown",
            ),
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter."""
        val = float(new_value)
        if parameter_path == "latency_sla":
            if val <= 0:
                raise ValueError("latency_sla must be positive")
            self.latency_sla = val
            return True
        elif parameter_path == "cpu_sla":
            if not (0.0 < val <= 100.0):
                raise ValueError("cpu_sla must be between 0 and 100")
            self.cpu_sla = val
            return True
        elif parameter_path == "confidence_target":
            if not (0.0 < val <= 1.0):
                raise ValueError("confidence_target must be between 0 and 1")
            self.confidence_target = val
            return True
        elif parameter_path == "cooldown_seconds":
            if val < 0:
                raise ValueError("cooldown_seconds must be >= 0")
            self.cooldown_seconds = val
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply dynamic configuration updates."""
        for param in ("latency_sla", "cpu_sla", "confidence_target", "cooldown_seconds"):
            if param in config:
                await self.update_parameter(param, config[param])

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return operational strategy metrics."""
        return {
            "total_assessments": float(self._total_assessments),
            "total_switches": float(self._total_switches),
            "sla_violations": float(self._sla_violations),
        }
