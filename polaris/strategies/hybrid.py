"""Hybrid strategy that delegates to multiple sub-strategies."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult, SystemState
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


class HybridStrategy(AdaptationStrategy):
    """
    Hybrid strategy that combines multiple strategies.

    Can delegate to multiple strategies and select the best action
    based on different selection modes.
    """

    def __init__(
        self,
        # (strategy, priority)
        strategies: List[Tuple[AdaptationStrategy, float]],
        selection_mode: str = "confidence",
        min_confidence: float = 0.7,
        cooldown_seconds: int = 0,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """
        Initialize hybrid strategy.

        Args:
            strategies: List of (strategy, priority) tuples
            selection_mode: How to select among proposals
                - 'first': Use first strategy that proposes action
                - 'priority': Use highest priority strategy
                - 'confidence': Use highest confidence action
            min_confidence: Minimum confidence threshold
            cooldown_seconds: Minimum seconds between any selected actions.
                When > 0 the entire assess() is skipped (returning []) if the
                cooldown has not elapsed since the last selected action.
                Default 0 means no cooldown.
        """
        self.strategies = sorted(strategies, key=lambda x: x[1], reverse=True)
        self.selection_mode = selection_mode
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self._last_action_time: Optional[datetime] = None
        self._adaptation_count = 0
        self._success_count = 0
        self._strategy_usage = dict.fromkeys(range(len(strategies)), 0)
        self._logger = logger
        self._metrics = metrics or NullMetricsCollector()

        if self._logger:
            self._logger.info(
                "HybridStrategy initialized",
                strategy_count=len(self.strategies),
                selection_mode=self.selection_mode,
                min_confidence=self.min_confidence,
            )

        self._metrics.increment("polaris.strategy.hybrid.initialized")

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> List[AdaptationAction]:
        """Assess using all strategies and select best action."""
        if self._logger:
            self._logger.debug(
                "HybridStrategy assessment started",
                system_id=state.system_id,
                selection_mode=self.selection_mode,
            )

        self._metrics.increment(
            "polaris.strategy.hybrid.assessments",
            tags={"system_id": state.system_id, "selection_mode": self.selection_mode},
        )

        now = datetime.now(timezone.utc)
        assess_start = now

        def _agentic_in_cooldown() -> bool:
            """Return True if the agentic (non-threshold) cooldown is still active."""
            if self.cooldown_seconds <= 0 or self._last_action_time is None:
                return False
            elapsed = (now - self._last_action_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                if self._logger:
                    self._logger.debug(
                        "HybridStrategy agentic cooldown active — skipping LLM strategies",
                        system_id=state.system_id,
                        remaining_seconds=round(remaining, 1),
                    )
                self._metrics.increment(
                    "polaris.strategy.hybrid.cooldown_skips",
                    tags={"system_id": state.system_id},
                )
                return True
            return False

        proposals = []

        if self.selection_mode == "first":
            # Sequential short-circuit: evaluate strategies in priority order and stop
            # as soon as one produces actions. Threshold (index 0) always runs; LLM
            # strategies (index > 0) are skipped while their cooldown is active.
            for i, (strategy, priority) in enumerate(self.strategies):
                if i > 0 and _agentic_in_cooldown():
                    break  # cooldown active — don't run any remaining LLM strategies
                try:
                    result = await strategy.assess(state, context)
                except Exception:
                    continue
                if isinstance(result, list) and result:
                    try:
                        conf_val = float(
                            await self._estimate_confidence(strategy, result[0], state)
                        )
                    except Exception:
                        conf_val = 0.7
                    proposals.append((result, conf_val, priority, i))
                    break  # first match wins; skip remaining strategies
        else:
            # Concurrent evaluation for priority/confidence modes.
            # Threshold (index 0) always runs; LLM strategies (index > 0) are skipped
            # while their cooldown is active.
            in_cooldown = _agentic_in_cooldown()
            tasks = []
            task_indices: List[int] = []
            for i, (strategy, _priority) in enumerate(self.strategies):
                if i > 0 and in_cooldown:
                    continue
                tasks.append(strategy.assess(state, context))
                task_indices.append(i)

            results: List[Union[List[AdaptationAction], BaseException]] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            confidence_tasks = []
            valid_indices = []
            for task_pos, result in enumerate(results):  # type: ignore[assignment]
                i = task_indices[task_pos]
                strategy, priority = self.strategies[i]
                if isinstance(result, Exception):  # type: ignore[unreachable]
                    continue  # type: ignore[unreachable]
                if isinstance(result, list) and result:
                    confidence_tasks.append(self._estimate_confidence(strategy, result[0], state))
                    valid_indices.append((i, priority, result, strategy))

            confidences: List[Union[float, BaseException]] = []
            if confidence_tasks:
                confidences = await asyncio.gather(*confidence_tasks, return_exceptions=True)

            for (i, priority, action_list, _strategy), conf in zip(valid_indices, confidences):
                if isinstance(conf, Exception):
                    conf_val = 0.7
                elif isinstance(conf, (int, float)):
                    conf_val = float(conf)
                else:
                    conf_val = 0.7
                proposals.append((action_list, conf_val, priority, i))

        if not proposals:
            if self._logger:
                self._logger.debug(
                    "HybridStrategy found no valid proposals",
                    system_id=state.system_id,
                )
            self._metrics.increment(
                "polaris.strategy.hybrid.selection_none",
                tags={"system_id": state.system_id, "mode": self.selection_mode},
            )
            duration = (datetime.now(timezone.utc) - assess_start).total_seconds()
            self._metrics.histogram(
                "polaris.strategy.hybrid.assess_duration_seconds",
                duration,
                tags={"system_id": state.system_id},
            )
            return []

        # Select based on mode
        selected = None
        selected_idx = None
        best_idx = None  # This variable is introduced by the provided snippet

        if self.selection_mode == "first":
            # Return first proposal (highest priority)
            selected, _, _, selected_idx = proposals[0]
            best_idx = selected_idx

        elif self.selection_mode == "priority":
            # Use highest priority strategy with valid action
            for action_list, conf, _pri, idx in sorted(proposals, key=lambda x: x[2], reverse=True):
                if conf >= self.min_confidence:
                    selected = action_list
                    selected_idx = idx
                    best_idx = selected_idx
                    break

        elif self.selection_mode == "confidence":
            # Return highest confidence proposal above threshold
            valid = [(al, c, p, i) for al, c, p, i in proposals if c >= self.min_confidence]
            if valid:
                selected, _, _, selected_idx = max(valid, key=lambda x: x[1])
                best_idx = selected_idx

        # Track which strategy was used.
        # Only update the agentic cooldown timer when a non-threshold (index > 0) strategy fires.
        if selected and selected_idx is not None:
            self._strategy_usage[selected_idx] += 1
            if selected_idx > 0:
                self._last_action_time = datetime.now(timezone.utc)

        self._metrics.histogram(
            "polaris.strategy.hybrid.assess_duration_seconds",
            (datetime.now(timezone.utc) - assess_start).total_seconds(),
            tags={"system_id": state.system_id},
        )
        if selected:
            self._metrics.increment(
                "polaris.strategy.hybrid.selection_success",
                tags={
                    "system_id": state.system_id,
                    "mode": self.selection_mode,
                    "strategy_index": str(best_idx),
                },
            )

        if self._logger:
            self._logger.debug(
                "HybridStrategy assessment completed",
                system_id=state.system_id,
                selected=bool(selected),
                action_count=len(selected) if selected else 0,
                selection_mode=self.selection_mode,
            )

        return selected if selected else []

    async def _estimate_confidence(
        self, strategy: AdaptationStrategy, action: AdaptationAction, state: SystemState
    ) -> float:
        """
        Estimate confidence in an action using strategy metrics when available.

        Fallback to a conservative default when metrics are unavailable.
        """
        # Default confidence
        base = 0.7
        try:
            metrics = await strategy.get_performance_metrics()
            if not isinstance(metrics, dict):
                return base  # type: ignore[unreachable]

            # Map success_rate (0..1) to confidence with slight shrinkage to avoid overconfidence
            sr = float(metrics.get("success_rate", base))
            sr = max(0.0, min(1.0, sr))
            confidence = 0.6 + 0.4 * sr  # range [0.6, 1.0]
            return confidence
        except Exception:
            return base

    async def on_action_executed(self, action: AdaptationAction, result: ExecutionResult) -> None:
        """Track adaptation success."""
        self._adaptation_count += 1
        if hasattr(result, "status") and result.status.value == "success":
            self._success_count += 1

        # Propagate to all strategies
        for strategy, _ in self.strategies:
            await strategy.on_action_executed(action, result)

    def get_tunable_parameters(self) -> Dict[str, ParameterSpec]:
        """Aggregate parameters from all sub-strategies."""
        params = {}

        # Add parameters from each strategy
        for i, (strategy, _) in enumerate(self.strategies):
            strategy_params = strategy.get_tunable_parameters()
            for path, spec in strategy_params.items():
                params[f"strategy_{i}.{path}"] = spec

        # Add hybrid-specific parameters
        params["selection_mode"] = ParameterSpec(
            current_value=self.selection_mode,
            type=str,
            allowed_values=["first", "priority", "confidence"],
            description="How to select between multiple strategy proposals",
            kind="selection_mode",
        )
        params["min_confidence"] = ParameterSpec(
            current_value=self.min_confidence,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence threshold for action selection",
            kind="confidence_threshold",
        )
        params["cooldown_seconds"] = ParameterSpec(
            current_value=self.cooldown_seconds,
            type=int,
            min_value=0,
            max_value=3600,
            description="Minimum seconds between any selected hybrid actions",
            kind="cooldown",
        )

        return params

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Route parameter updates to appropriate sub-strategy."""
        if parameter_path.startswith("strategy_"):
            # Parse strategy index and delegate
            parts = parameter_path.split(".", 1)
            strategy_idx = int(parts[0].split("_")[1])
            sub_path = parts[1] if len(parts) > 1 else ""

            if strategy_idx < len(self.strategies):
                return await self.strategies[strategy_idx][0].update_parameter(sub_path, new_value)

        elif parameter_path == "selection_mode":
            if new_value in ["first", "priority", "confidence"]:
                self.selection_mode = new_value
                return True

        elif parameter_path == "min_confidence":
            self.min_confidence = float(new_value)
            return True

        elif parameter_path == "cooldown_seconds":
            self.cooldown_seconds = int(new_value)
            return True

        return False

    async def apply_config_update(self, config: Dict[str, Any]) -> None:
        """Apply configuration updates to the hybrid strategy."""
        if "selection_mode" in config:
            await self.update_parameter("selection_mode", config["selection_mode"])
        if "min_confidence" in config:
            await self.update_parameter("min_confidence", config["min_confidence"])
        if "cooldown_seconds" in config:
            await self.update_parameter("cooldown_seconds", config["cooldown_seconds"])

        new_subs = config.get("strategies", [])
        if isinstance(new_subs, list) and len(new_subs) == len(self.strategies):
            for sub_conf, (sub_strategy, _prio) in zip(new_subs, self.strategies):
                if not isinstance(sub_conf, dict):
                    continue
                s_type = sub_conf.get("type")
                if s_type == "threshold":
                    th = sub_conf.get("threshold", {}) or {}
                    cd = th.get("cooldown_seconds")
                    if cd is not None and hasattr(sub_strategy, "update_parameter"):
                        await sub_strategy.update_parameter("cooldown_seconds", cd)
                    thresh = th.get("thresholds", {}) or {}
                    for metric, vals in thresh.items():
                        if not isinstance(vals, dict):
                            continue
                        if "high" in vals:
                            await sub_strategy.update_parameter(
                                f"thresholds.{metric}.high", vals["high"]
                            )
                        if "low" in vals:
                            await sub_strategy.update_parameter(
                                f"thresholds.{metric}.low", vals["low"]
                            )
                elif s_type == "llm_reasoning":
                    llm_cfg = sub_conf.get("llm_reasoning", {}) or {}
                    if "temperature" in llm_cfg and hasattr(sub_strategy, "update_parameter"):
                        await sub_strategy.update_parameter("temperature", llm_cfg["temperature"])
                    if "system_description" in llm_cfg and hasattr(
                        sub_strategy, "update_parameter"
                    ):
                        await sub_strategy.update_parameter(
                            "system_description", llm_cfg["system_description"]
                        )
                    resil = llm_cfg.get("resilience")
                    if (
                        resil
                        and hasattr(sub_strategy, "llm")
                        and hasattr(sub_strategy.llm, "update_resilience")
                    ):
                        try:
                            sub_strategy.llm.update_resilience(resil)
                        except Exception as e:
                            if self._logger:
                                self._logger.warning(
                                    "Failed to hot-update sub-strategy LLM resilience",
                                    error=str(e),
                                )

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Return strategy performance metrics."""
        metrics = {}

        if self._adaptation_count > 0:
            metrics["success_rate"] = self._success_count / self._adaptation_count
            metrics["total_adaptations"] = float(self._adaptation_count)

        # Add usage statistics
        for idx, count in self._strategy_usage.items():
            metrics[f"strategy_{idx}_usage"] = float(count)

        return metrics
