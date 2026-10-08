"""SUAVE-specific threshold strategy implementation."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


class SuaveThresholdStrategy(AdaptationStrategy):
    """
    Threshold strategy specialized for SUAVE R1/R2 mode switching.

    Behavior
    --------
        - Only triggers under degraded conditions ("too horrible"):
            - visibility below trigger threshold, or
            - thruster failure detected.
        - This is the reactive baseline described in the paper: it reacts to
            current visibility and thruster status, while the agentic strategy can
            use higher-level reasoning and trends to act proactively.
    - When triggered, emits two ``change_mode`` actions:
      - R1: search-path mode from visibility bands
      - R2: maintain-motion mode from thruster failure state
    """

    def __init__(
        self,
        visibility_metric_names: Optional[Sequence[str]] = None,
        thruster_failure_metric_names: Optional[Sequence[str]] = None,
        performance_metric_names: Optional[Sequence[str]] = None,
        trigger_visibility_below: float = 1.0,
        trigger_performance_at_or_above: float = 1.0,
        trigger_thruster_failure_at_or_above: float = 0.5,
        visibility_medium_at_or_above: float = 1.0,
        visibility_high_at_or_above: float = 2.0,
        search_path_function_node: str = "f_generate_search_path",
        maintain_motion_function_node: str = "f_maintain_motion",
        spiral_low_mode: str = "fd_spiral_low",
        spiral_medium_mode: str = "fd_spiral_medium",
        spiral_high_mode: str = "fd_spiral_high",
        recover_thrusters_mode: str = "fd_recover_thrusters",
        all_thrusters_mode: str = "fd_all_thrusters",
        cooldown_seconds: int = 0,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize SUAVE threshold strategy with visibility and thruster parameters."""
        self.visibility_metric_names = list(
            visibility_metric_names
            or [
                "water_visibility",
                "water_visibility_observer.value",
                "visibility",
            ]
        )
        self.thruster_failure_metric_names = list(
            thruster_failure_metric_names
            or [
                "thruster_failure_detected",
                "thruster_failure",
                "thruster_failed",
            ]
        )
        # Performance metrics are retained for observability and optional
        # future extensions, but the paper's baseline reactive strategy uses
        # visibility and thruster status as its actual triggers.
        self.performance_metric_names = list(
            performance_metric_names
            or [
                "diagnostics.error_count",
                "performance_error_count",
            ]
        )

        self.trigger_visibility_below = float(trigger_visibility_below)
        self.trigger_performance_at_or_above = float(trigger_performance_at_or_above)
        self.trigger_thruster_failure_at_or_above = float(trigger_thruster_failure_at_or_above)

        self.visibility_medium_at_or_above = float(visibility_medium_at_or_above)
        self.visibility_high_at_or_above = float(visibility_high_at_or_above)

        self.search_path_function_node = search_path_function_node
        self.maintain_motion_function_node = maintain_motion_function_node
        self.spiral_low_mode = spiral_low_mode
        self.spiral_medium_mode = spiral_medium_mode
        self.spiral_high_mode = spiral_high_mode
        self.recover_thrusters_mode = recover_thrusters_mode
        self.all_thrusters_mode = all_thrusters_mode

        self.cooldown_seconds = int(max(0, cooldown_seconds))
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()

        self._last_adaptation: Dict[str, datetime] = {}
        self._adaptation_count = 0
        self._success_count = 0

        if self.logger:
            self.logger.info(
                "SUAVE threshold strategy initialized",
                visibility_metric_names=self.visibility_metric_names,
                thruster_failure_metric_names=self.thruster_failure_metric_names,
                performance_metric_names=self.performance_metric_names,
                trigger_visibility_below=self.trigger_visibility_below,
                trigger_performance_at_or_above=self.trigger_performance_at_or_above,
                trigger_thruster_failure_at_or_above=self.trigger_thruster_failure_at_or_above,
                cooldown_seconds=self.cooldown_seconds,
            )

        self.metrics.increment("polaris.strategy.suave_threshold.initialized")

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess SUAVE state and emit R1/R2 change_mode actions when triggered."""
        self.metrics.increment(
            "polaris.strategy.suave_threshold.assessments", tags={"system_id": state.system_id}
        )

        now = datetime.now(timezone.utc)
        last = self._last_adaptation.get(state.system_id)
        if last and (now - last).total_seconds() < self.cooldown_seconds:
            self.metrics.increment(
                "polaris.strategy.suave_threshold.cooldown_blocked",
                tags={"system_id": state.system_id},
            )
            return []

        visibility = self._extract_metric_float(state, self.visibility_metric_names)
        thruster_raw = self._extract_metric_float(state, self.thruster_failure_metric_names)
        thruster_failed = (
            thruster_raw is not None and thruster_raw >= self.trigger_thruster_failure_at_or_above
        )
        visibility_bad = visibility is not None and visibility < self.trigger_visibility_below

        if not (thruster_failed or visibility_bad):
            self.metrics.increment(
                "polaris.strategy.suave_threshold.no_action_needed",
                tags={"system_id": state.system_id},
            )
            return []

        # Preserve performance as an informational metric only, so callers can
        # inspect it in logs without it changing the paper's reactive baseline.
        performance = self._extract_metric_float(state, self.performance_metric_names)

        visibility_mode = self._visibility_to_mode(visibility)
        motion_mode = self.recover_thrusters_mode if thruster_failed else self.all_thrusters_mode

        actions = [
            AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type="change_mode",
                target_system=state.system_id,
                parameters={
                    "function_node": self.search_path_function_node,
                    "mode": visibility_mode,
                },
            ),
            AdaptationAction(
                action_id=str(uuid.uuid4()),
                action_type="change_mode",
                target_system=state.system_id,
                parameters={
                    "function_node": self.maintain_motion_function_node,
                    "mode": motion_mode,
                },
            ),
        ]

        self._last_adaptation[state.system_id] = now
        self.metrics.increment(
            "polaris.strategy.suave_threshold.actions_proposed",
            tags={
                "system_id": state.system_id,
                "visibility_mode": visibility_mode,
                "motion_mode": motion_mode,
            },
        )

        if self.logger:
            self.logger.info(
                "SUAVE threshold strategy triggered",
                system_id=state.system_id,
                visibility=visibility,
                performance=performance,
                thruster_failed=thruster_failed,
                visibility_bad=visibility_bad,
                visibility_mode=visibility_mode,
                motion_mode=motion_mode,
            )

        return actions

    def _extract_metric_float(self, state: SystemState, names: Sequence[str]) -> Optional[float]:
        for metric_name in names:
            metric = state.metrics.get(metric_name)
            if metric is None:
                continue
            try:
                return float(metric.value)
            except (TypeError, ValueError):
                continue
        return None

    def _visibility_to_mode(self, visibility: Optional[float]) -> str:
        if visibility is None:
            return self.spiral_low_mode
        if visibility >= self.visibility_high_at_or_above:
            return self.spiral_high_mode
        if visibility >= self.visibility_medium_at_or_above:
            return self.spiral_medium_mode
        return self.spiral_low_mode

    async def on_action_executed(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Record metrics when an action is executed."""
        self._adaptation_count += 1
        if result.status.value == "success":
            self._success_count += 1

        self.metrics.increment(
            "polaris.strategy.suave_threshold.actions_executed",
            tags={
                "action_type": action.action_type,
                "system_id": action.target_system,
                "status": result.status.value,
            },
        )

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Return the tunable parameters for SUAVE threshold strategy."""
        return {
            "trigger_visibility_below": ParameterSpec(
                current_value=self.trigger_visibility_below,
                type=float,
                min_value=0.0,
                max_value=10.0,
                description="Trigger adaptation when visibility is below this threshold",
            ),
            "trigger_performance_at_or_above": ParameterSpec(
                current_value=self.trigger_performance_at_or_above,
                type=float,
                min_value=0.0,
                max_value=100.0,
                description="Trigger adaptation when performance metric reaches this threshold",
            ),
            "trigger_thruster_failure_at_or_above": ParameterSpec(
                current_value=self.trigger_thruster_failure_at_or_above,
                type=float,
                min_value=0.0,
                max_value=1.0,
                description="Threshold used to interpret thruster failure metric as failed",
            ),
            "visibility_medium_at_or_above": ParameterSpec(
                current_value=self.visibility_medium_at_or_above,
                type=float,
                min_value=0.0,
                max_value=10.0,
                description="Visibility threshold for medium spiral mode",
            ),
            "visibility_high_at_or_above": ParameterSpec(
                current_value=self.visibility_high_at_or_above,
                type=float,
                min_value=0.0,
                max_value=10.0,
                description="Visibility threshold for high spiral mode",
            ),
            "cooldown_seconds": ParameterSpec(
                current_value=self.cooldown_seconds,
                type=int,
                min_value=0,
                max_value=3600,
                description="Minimum seconds between SUAVE threshold actions",
                kind="cooldown",
            ),
        }

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Update a tunable parameter value."""
        if parameter_path == "trigger_visibility_below":
            self.trigger_visibility_below = float(new_value)
            return True
        if parameter_path == "trigger_performance_at_or_above":
            self.trigger_performance_at_or_above = float(new_value)
            return True
        if parameter_path == "trigger_thruster_failure_at_or_above":
            self.trigger_thruster_failure_at_or_above = float(new_value)
            return True
        if parameter_path == "visibility_medium_at_or_above":
            self.visibility_medium_at_or_above = float(new_value)
            return True
        if parameter_path == "visibility_high_at_or_above":
            self.visibility_high_at_or_above = float(new_value)
            return True
        if parameter_path == "cooldown_seconds":
            self.cooldown_seconds = int(max(0, int(new_value)))
            return True
        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to strategy parameters."""
        for key in [
            "trigger_visibility_below",
            "trigger_performance_at_or_above",
            "trigger_thruster_failure_at_or_above",
            "visibility_medium_at_or_above",
            "visibility_high_at_or_above",
            "cooldown_seconds",
        ]:
            if key in config:
                await self.update_parameter(key, config[key])

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the strategy."""
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}
        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }
