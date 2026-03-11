"""Threshold reactive strategy implementation."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState
from polaris.infrastructure.constants import DEFAULT_COOLDOWN_SECONDS
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


class ThresholdReactiveStrategy(AdaptationStrategy):
    """
    Simple threshold-based reactive strategy.

    Triggers adaptations when metric values cross defined thresholds.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize threshold strategy.

        Args:
            thresholds: Dict of {metric: {'high': value, 'low': value}}
            cooldown_seconds: Minimum time between adaptations
            logger: Logger instance for structured logging
            metrics: Metrics collector for tracking strategy performance
        """
        self.thresholds = thresholds or {
            "cpu_usage": {"high": 80.0, "low": 20.0},
            "memory_usage": {"high": 85.0, "low": 25.0},
        }
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._last_adaptation: Dict[str, datetime] = {}
        self._adaptation_count = 0
        self._success_count = 0

        if self.logger:
            self.logger.info(
                "Threshold strategy initialized",
                thresholds=self.thresholds,
                cooldown_seconds=cooldown_seconds,
            )

        self.metrics.increment("polaris.strategy.threshold.initialized")
        self.metrics.gauge("polaris.strategy.threshold.cooldown_seconds", cooldown_seconds)

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Check if any thresholds are crossed."""
        if self.logger:
            self.logger.debug(
                "Assessing thresholds for system",
                system_id=state.system_id,
                metric_count=len(state.metrics),
            )

        self.metrics.increment(
            "polaris.strategy.threshold.assessments", tags={"system_id": state.system_id}
        )

        # Check cooldown
        now = datetime.now(timezone.utc)
        last = self._last_adaptation.get(state.system_id)
        if last and (now - last).total_seconds() < self.cooldown_seconds:
            remaining_cooldown = self.cooldown_seconds - (now - last).total_seconds()
            if self.logger:
                self.logger.debug(
                    "System in cooldown period",
                    system_id=state.system_id,
                    remaining_seconds=remaining_cooldown,
                )
            self.metrics.increment(
                "polaris.strategy.threshold.cooldown_blocked",
                tags={"system_id": state.system_id},
            )
            return []  # Still in cooldown

        # Check each metric against thresholds
        for metric_name, metric in state.metrics.items():
            if metric_name not in self.thresholds:
                if self.logger:
                    self.logger.debug(
                        "No threshold configured for metric",
                        metric=metric_name,
                        system_id=state.system_id,
                    )
                continue

            try:
                value = float(metric.value)
                thresholds = self.thresholds[metric_name]

                if self.logger:
                    self.logger.debug(
                        "Evaluating metric against thresholds",
                        metric=metric_name,
                        value=value,
                        thresholds=thresholds,
                        system_id=state.system_id,
                    )

                self.metrics.histogram(
                    "polaris.strategy.threshold.metric_values",
                    value,
                    tags={"metric": metric_name, "system_id": state.system_id},
                )

                # Check if threshold crossed
                if "high" in thresholds and value > thresholds["high"]:
                    if self.logger:
                        self.logger.info(
                            "High threshold exceeded, creating scale action",
                            metric=metric_name,
                            value=value,
                            threshold=thresholds["high"],
                            system_id=state.system_id,
                        )

                    self.metrics.increment(
                        "polaris.strategy.threshold.high_threshold_exceeded",
                        tags={"metric": metric_name, "system_id": state.system_id},
                    )

                    action = self._create_scale_action(
                        state.system_id, metric_name, value, "high", thresholds["high"]
                    )
                    self._last_adaptation[state.system_id] = now
                    return [action]

                elif "low" in thresholds and value < thresholds["low"]:
                    if self.logger:
                        self.logger.info(
                            "Low threshold breached, creating scale action",
                            metric=metric_name,
                            value=value,
                            threshold=thresholds["low"],
                            system_id=state.system_id,
                        )

                    self.metrics.increment(
                        "polaris.strategy.threshold.low_threshold_breached",
                        tags={"metric": metric_name, "system_id": state.system_id},
                    )

                    action = self._create_scale_action(
                        state.system_id, metric_name, value, "low", thresholds["low"]
                    )
                    self._last_adaptation[state.system_id] = now
                    return [action]

            except (ValueError, TypeError) as e:
                if self.logger:
                    self.logger.warning(
                        "Failed to parse metric value",
                        metric=metric_name,
                        value=metric.value,
                        error=str(e),
                        system_id=state.system_id,
                    )
                self.metrics.increment(
                    "polaris.strategy.threshold.no_action_needed",
                    tags={"system_id": state.system_id},
                )
                continue

        if self.logger:
            self.logger.debug("No thresholds exceeded", system_id=state.system_id)
        self.metrics.increment(
            "polaris.strategy.threshold.no_action_needed", tags={"system_id": state.system_id}
        )
        return []

    def _create_scale_action(
        self, system_id: str, metric: str, value: float, threshold_type: str, threshold_value: float
    ) -> AdaptationAction:
        """Create a scale action based on threshold crossing."""
        # For server_count, the logic is inverted:
        # - Low server count -> scale up
        # - High server count -> scale down
        if metric == "server_count":
            action_type = "scale_up" if threshold_type == "low" else "scale_down"
        else:
            # For other metrics (CPU, memory, response time):
            # - High values -> scale up
            # - Low values -> scale down
            action_type = "scale_up" if threshold_type == "high" else "scale_down"

        return AdaptationAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            target_system=system_id,
            parameters={
                "metric": metric,
                "current_value": value,
                "threshold": threshold_value,
                "instances": 1,
            },
        )

    async def on_action_executed(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Track adaptation success."""
        self._adaptation_count += 1
        if hasattr(result, "status") and result.status.value == "success":
            self._success_count += 1

        self.metrics.increment(
            "polaris.strategy.threshold.actions_executed",
            tags={
                "action_type": action.action_type,
                "system_id": action.target_system,
                "status": result.status.value if hasattr(result, "status") else "unknown",
            },
        )
        self.metrics.gauge(
            "polaris.strategy.threshold.success_rate",
            self._success_count / self._adaptation_count if self._adaptation_count > 0 else 0,
        )

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Return tunable parameters."""
        params = {}

        # Threshold values
        for metric, thresholds in self.thresholds.items():
            if "high" in thresholds:
                current_high = thresholds["high"]
                params[f"thresholds.{metric}.high"] = ParameterSpec(
                    current_value=current_high,
                    type=float,
                    min_value=max(0.0, round(current_high * 0.25, 6)),
                    max_value=round(current_high * 1.75, 6),
                    description=f"High threshold for {metric}",
                    kind="threshold_high",
                )
            if "low" in thresholds:
                current_low = thresholds["low"]
                params[f"thresholds.{metric}.low"] = ParameterSpec(
                    current_value=current_low,
                    type=float,
                    min_value=max(0.0, round(current_low * 0.25, 6)),
                    max_value=round(current_low * 1.75, 6),
                    description=f"Low threshold for {metric}",
                    kind="threshold_low",
                )

        # Cooldown
        params["cooldown_seconds"] = ParameterSpec(
            current_value=self.cooldown_seconds,
            type=int,
            min_value=10,
            max_value=300,
            description="Minimum seconds between adaptations",
            kind="cooldown",
        )

        return params

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a parameter."""
        if parameter_path == "cooldown_seconds":
            self.cooldown_seconds = int(new_value)
            return True

        elif parameter_path.startswith("thresholds."):
            parts = parameter_path.split(".")
            if len(parts) == 3:
                metric, threshold_type = parts[1], parts[2]
                if metric in self.thresholds:
                    self.thresholds[metric][threshold_type] = float(new_value)
                    return True

        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to the threshold strategy."""
        cooldown = config.get("cooldown_seconds")
        if cooldown is not None:
            await self.update_parameter("cooldown_seconds", cooldown)

        thresholds = config.get("thresholds", {}) or {}
        for metric, vals in thresholds.items():
            if not isinstance(vals, dict):
                continue
            if "high" in vals:
                await self.update_parameter(f"thresholds.{metric}.high", vals["high"])
            if "low" in vals:
                await self.update_parameter(f"thresholds.{metric}.low", vals["low"])

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}

        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }
