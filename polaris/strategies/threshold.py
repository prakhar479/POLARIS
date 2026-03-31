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
    """Simple threshold-based reactive strategy.

    Triggers adaptations when metric values cross defined thresholds.

    Opt into hybrid cooldown exemption so that when used inside a
    ``HybridStrategy`` this strategy continues to evaluate during cooldown
    while heavier agentic strategies are paused.
    """

    # Allow HybridStrategy to identify this as a lightweight guard strategy
    # that should bypass its cooldown logic.  Custom strategies may set this
    # attribute to True to achieve the same effect.
    hybrid_cooldown_exempt: bool = True

    def __init__(
        self,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        action_templates: Optional[Dict[str, Any]] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize threshold strategy.

        Args:
            thresholds: Dict of {metric: {'high': value, 'low': value}}.
                Defaults to an empty mapping — if not provided no actions will
                ever be triggered.  You must configure at least one metric
                threshold for the strategy to be useful.
            action_templates: Per-metric action templates for threshold
                crossings.  Expected shape::

                    {
                        "default": {
                            "high": {"type": "scale_up", "parameters": {}},
                            "low":  {"type": "scale_down", "parameters": {}},
                        },
                        "<metric_name>": {
                            "high": {"type": "...", "parameters": {}},
                        },
                    }

                A ``"default"`` entry acts as a fallback for metrics that do
                not have their own template.  Must be provided when
                ``thresholds`` is non-empty.
            cooldown_seconds: Minimum time between adaptations
            logger: Logger instance for structured logging
            metrics: Metrics collector for tracking strategy performance
        """
        self.thresholds = thresholds or {}
        self.action_templates = self._normalize_action_templates(action_templates)
        self.cooldown_seconds = cooldown_seconds
        self.logger = logger
        self.metrics = metrics or NullMetricsCollector()
        self._last_adaptation: Dict[str, datetime] = {}
        self._adaptation_count = 0
        self._success_count = 0

        if self.thresholds and not self.action_templates:
            if self.logger:
                self.logger.warning(
                    "ThresholdReactiveStrategy: thresholds configured but no "
                    "action_templates provided — threshold crossings will raise an error."
                )

        if self.logger:
            self.logger.info(
                "Threshold strategy initialized",
                thresholds=self.thresholds,
                action_templates=self.action_templates,
                cooldown_seconds=cooldown_seconds,
            )

        self.metrics.increment("polaris.strategy.threshold.initialized")
        self.metrics.gauge("polaris.strategy.threshold.cooldown_seconds", cooldown_seconds)

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Check if any thresholds are crossed."""
        if not self.thresholds:
            # No thresholds configured — nothing to evaluate. This is not an
            # error; the strategy may be intentionally unconfigured (e.g. as a
            # placeholder in a hybrid setup).
            return []

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
                    "polaris.strategy.threshold.metric_parse_errors",
                    tags={"metric": metric_name, "system_id": state.system_id},
                )
                continue

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

            # Check if threshold crossed; ValueError from missing template propagates up
            if "high" in thresholds and value > thresholds["high"]:
                if self.logger:
                    self.logger.info(
                        "High threshold exceeded, creating action",
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
                        "Low threshold breached, creating action",
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

        if self.logger:
            self.logger.debug("No thresholds exceeded", system_id=state.system_id)
        self.metrics.increment(
            "polaris.strategy.threshold.no_action_needed", tags={"system_id": state.system_id}
        )
        return []

    def _create_scale_action(
        self, system_id: str, metric: str, value: float, threshold_type: str, threshold_value: float
    ) -> AdaptationAction:
        """Create an action for a threshold crossing using configured templates."""
        template = self._resolve_action_template(metric, threshold_type)
        action_type = template["type"]
        template_parameters = template.get("parameters", {})
        parameters = dict(template_parameters) if isinstance(template_parameters, dict) else {}
        parameters["metric"] = metric
        parameters["current_value"] = value
        parameters["threshold"] = threshold_value
        parameters.setdefault("instances", 1)

        return AdaptationAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            target_system=system_id,
            parameters=parameters,
        )

    def _normalize_action_templates(
        self, action_templates: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Normalize user-provided action templates into a canonical structure.

        Only entries that have a valid ``type`` string are kept.  No default
        actions are injected — all behaviour must be configured explicitly.
        """
        templates: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if not isinstance(action_templates, dict):
            return templates

        for metric, rules in action_templates.items():
            if not isinstance(metric, str) or not metric.strip() or not isinstance(rules, dict):
                continue

            metric_key = metric.strip()
            metric_templates: Dict[str, Dict[str, Any]] = {}
            for threshold_type in ("high", "low"):
                candidate = rules.get(threshold_type)
                if not isinstance(candidate, dict):
                    continue
                action_type = candidate.get("type")
                if not isinstance(action_type, str) or not action_type.strip():
                    continue
                parameters = candidate.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                metric_templates[threshold_type] = {
                    "type": action_type.strip(),
                    "parameters": dict(parameters),
                }

            if metric_templates:
                templates[metric_key] = metric_templates

        return templates

    def _resolve_action_template(self, metric: str, threshold_type: str) -> Dict[str, Any]:
        """Resolve action template for a metric and threshold direction.

        Lookup order:
        1. Per-metric template (``action_templates[metric][threshold_type]``)
        2. Default fallback template (``action_templates["default"][threshold_type]``)

        Raises:
            ValueError: If no template is found for the metric/direction combination.
                This indicates a configuration error — the user has configured a
                threshold without a corresponding action template.
        """
        metric_rules = self.action_templates.get(metric, {})
        template = metric_rules.get(threshold_type)
        if isinstance(template, dict) and isinstance(template.get("type"), str):
            return template

        default_rules = self.action_templates.get("default", {})
        default_template = default_rules.get(threshold_type)
        if isinstance(default_template, dict) and isinstance(default_template.get("type"), str):
            return default_template

        raise ValueError(
            f"ThresholdReactiveStrategy: no action_template configured for "
            f"metric='{metric}' threshold_type='{threshold_type}'. "
            f"Add a per-metric or 'default' entry to action_templates."
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

        if "action_templates" in config and isinstance(config["action_templates"], dict):
            self.action_templates = self._normalize_action_templates(config["action_templates"])

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        if self._adaptation_count == 0:
            return {"success_rate": 0.0}

        return {
            "success_rate": self._success_count / self._adaptation_count,
            "total_adaptations": float(self._adaptation_count),
        }
