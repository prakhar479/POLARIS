"""Hybrid strategy that delegates to multiple sub-strategies."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.strategy import AdaptationContext, AdaptationStrategy, ParameterSpec
from polaris.core.models import AdaptationAction, ExecutionResult, HealthStatus, SystemState
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector


class HybridStrategy(AdaptationStrategy):
    """Hybrid strategy that combines multiple strategies.

    Can delegate to multiple strategies and select the best action based on different
    selection modes.
    """

    def __init__(
        self,
        # (strategy, priority)
        strategies: List[Tuple[AdaptationStrategy, float]],
        selection_mode: str = "confidence",
        min_confidence: float = 0.7,
        cooldown_seconds: int = 0,
        objective_weights: Optional[Dict[str, float]] = None,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize hybrid strategy.

        Args:
            strategies: List of (strategy, priority) tuples
            selection_mode: How to select among proposals - 'first': Use first strategy
                that proposes action - 'priority': Use highest priority strategy -
                'confidence': Use highest confidence action - 'pareto': Multi-objective
                Pareto utility ranking
            min_confidence: Minimum confidence threshold
            cooldown_seconds: Minimum seconds between cooldown-restricted selected
                actions (typically agentic/LLM-backed). Cooldown-exempt strategies
                continue to run while cooldown is active. Default 0 means no cooldown.
            objective_weights: Optional dictionary of weights for Pareto selection
                (defaults: performance: 0.5, cost: 0.3, qos: 0.2)
            logger: Optional logger for observability
            metrics: Optional metrics collector
        """
        self.strategies = sorted(strategies, key=lambda x: x[1], reverse=True)
        self.selection_mode = selection_mode
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.objective_weights = objective_weights or {
            "performance": 0.5,
            "cost": 0.3,
            "qos": 0.2,
        }
        self._last_action_time: Optional[datetime] = None
        self._adaptation_count = 0
        self._success_count = 0
        self._strategy_usage = dict.fromkeys(range(len(strategies)), 0)
        self._logger = logger
        self._metrics = metrics or NullMetricsCollector()
        self._cooldown_exempt_indices = {
            idx
            for idx, (strategy, _priority) in enumerate(self.strategies)
            if self._is_cooldown_exempt_strategy(strategy)
        }

        if self._logger:
            self._logger.info(
                "HybridStrategy initialized",
                strategy_count=len(self.strategies),
                selection_mode=self.selection_mode,
                min_confidence=self.min_confidence,
            )

        self._metrics.increment("polaris.strategy.hybrid.initialized")

    def _is_cooldown_exempt_strategy(self, strategy: AdaptationStrategy) -> bool:
        """Return True when a sub-strategy should bypass hybrid cooldown.

        Strategies opt in by setting ``hybrid_cooldown_exempt = True`` as a
        class attribute (or instance attribute). This avoids coupling to any
        specific strategy class name.
        """
        return bool(getattr(strategy, "hybrid_cooldown_exempt", False))

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
            # as soon as one produces actions. During cooldown, only cooldown-exempt
            # strategies are evaluated.
            in_cooldown = _agentic_in_cooldown()
            for i, (strategy, priority) in enumerate(self.strategies):
                if in_cooldown and i not in self._cooldown_exempt_indices:
                    continue
                try:
                    result = await strategy.assess(state, context)
                except Exception as exc:
                    if self._logger:
                        self._logger.warning(
                            "Sub-strategy assessment failed in hybrid selection_mode=first",
                            strategy_index=i,
                            error=str(exc),
                        )
                    self._metrics.increment(
                        "polaris.strategy.hybrid.sub_strategy_errors",
                        tags={"strategy_index": str(i)},
                    )
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
            # During cooldown, only cooldown-exempt strategies are evaluated.
            in_cooldown = _agentic_in_cooldown()
            tasks = []
            task_indices: List[int] = []
            for i, (strategy, _priority) in enumerate(self.strategies):
                if in_cooldown and i not in self._cooldown_exempt_indices:
                    continue
                tasks.append(strategy.assess(state, context))
                task_indices.append(i)

            results: List[Union[List[AdaptationAction], BaseException]] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            confidence_tasks = []
            valid_indices = []
            for task_pos, outcome in enumerate(results):
                i = task_indices[task_pos]
                strategy, priority = self.strategies[i]
                if isinstance(outcome, BaseException):
                    if self._logger:
                        self._logger.warning(
                            "Sub-strategy assessment failed in hybrid concurrent mode",
                            strategy_index=i,
                            error=str(outcome),
                        )
                    self._metrics.increment(
                        "polaris.strategy.hybrid.sub_strategy_errors",
                        tags={"strategy_index": str(i)},
                    )
                    continue
                if isinstance(outcome, list) and outcome:
                    confidence_tasks.append(self._estimate_confidence(strategy, outcome[0], state))
                    valid_indices.append((i, priority, outcome, strategy))

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

        if self.selection_mode == "first":
            # Return first proposal (highest priority)
            selected, _, _, selected_idx = proposals[0]

        elif self.selection_mode == "priority":
            # Use highest priority strategy with valid action
            for action_list, conf, _pri, idx in sorted(proposals, key=lambda x: x[2], reverse=True):
                if conf >= self.min_confidence:
                    selected = action_list
                    selected_idx = idx
                    break

        elif self.selection_mode == "confidence":
            # Return highest confidence proposal above threshold
            valid = [(al, c, p, i) for al, c, p, i in proposals if c >= self.min_confidence]
            if valid:
                selected, _, _, selected_idx = max(valid, key=lambda x: x[1])

        elif self.selection_mode in ("pareto", "multi_objective"):
            # Multi-objective Pareto utility selection
            valid = [(al, c, p, i) for al, c, p, i in proposals if c >= self.min_confidence]
            if valid:
                scored = []
                for al, c, p, i in valid:
                    act = al[0]
                    score = self._calculate_pareto_utility(act, c, state, context)
                    scored.append((al, score, p, i))
                selected, _, _, selected_idx = max(scored, key=lambda x: x[1])

        # Track which strategy was used.
        # Only update cooldown timestamp when a cooldown-restricted strategy fires.
        if selected and selected_idx is not None:
            self._strategy_usage[selected_idx] += 1
            if selected_idx not in self._cooldown_exempt_indices:
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
                    "strategy_index": str(selected_idx),
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
        """Estimate confidence in an action using strategy metrics when available.

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

    def _calculate_pareto_utility(
        self,
        action: AdaptationAction,
        confidence: float,
        state: SystemState,
        context: Optional[AdaptationContext] = None,
    ) -> float:
        """Calculate multi-objective utility score for a candidate action.

        Balancing:
        - Performance / SLA risk mitigation (higher is better)
        - Cost efficiency (higher is cheaper / less resource consumption)
        - Quality of Service (QoS) preservation

        Supports contract-driven ActionSchema impact hints and SLO contracts,
        with fallback to heuristic pattern matching for legacy exemplars.
        """
        action_type = (action.action_type or "").lower()

        # Check current system load/stress indicators
        high_load = getattr(state, "health_status", None) in (
            HealthStatus.WARNING,
            HealthStatus.CRITICAL,
        )

        # 1. Contract-driven SLO violations check
        contract = getattr(context, "system_contract", None) if context else None
        schema = None
        if contract is not None:
            if hasattr(contract, "get_action_schema"):
                schema = contract.get_action_schema(action.action_type)
            if not high_load and getattr(contract, "slos", None):
                for slo in contract.slos:
                    mv = state.metrics.get(slo.metric_name)
                    if mv is not None:
                        try:
                            if slo.is_violated(float(mv.value)):
                                high_load = True
                                break
                        except (TypeError, ValueError):
                            pass

        # 2. Heuristic metric fallback checks if high_load not already triggered
        if not high_load:
            for m_name in ("average_utilization", "cpu_usage", "utilization"):
                mv = state.metrics.get(m_name)
                if mv is not None:
                    try:
                        u_val = float(mv.value)
                        if u_val > 75.0 or (0.0 <= u_val <= 1.0 and u_val > 0.75):
                            high_load = True
                            break
                    except (TypeError, ValueError):
                        pass

        if not high_load:
            for m_name in ("average_response_time", "response_time", "latency"):
                mv = state.metrics.get(m_name)
                if mv is not None:
                    try:
                        val = float(mv.value)
                        if val > 500.0 or (0.75 < val < 10.0):
                            high_load = True
                            break
                    except (TypeError, ValueError):
                        pass

        # Check downstream peer health for cascade failure prevention
        downstream_stressed = False
        if context and context.peer_states and getattr(context, "downstream_systems", None):
            for ds_id in context.downstream_systems:
                peer_state = context.peer_states.get(ds_id)
                if peer_state is not None:
                    if getattr(peer_state, "health_status", None) in (
                        HealthStatus.WARNING,
                        HealthStatus.CRITICAL,
                    ):
                        downstream_stressed = True
                        break
                    for m_name in ("average_utilization", "cpu_usage", "utilization"):
                        pmv = peer_state.metrics.get(m_name)
                        if pmv is not None:
                            try:
                                pu_val = float(pmv.value)
                                if pu_val > 80.0 or (0.0 <= pu_val <= 1.0 and pu_val > 0.80):
                                    downstream_stressed = True
                                    break
                            except (TypeError, ValueError):
                                pass

        # 1. Performance utility
        if schema and getattr(schema, "performance_impact", "neutral") != "neutral":
            if schema.performance_impact == "positive":
                u_perf = 1.0 if high_load else 0.6
            else:
                u_perf = 0.1 if high_load else 0.7
        elif "scale_up" in action_type:
            u_perf = 1.0 if high_load else 0.6
        elif "scale_down" in action_type:
            u_perf = 0.1 if high_load else 0.7
        elif "dimmer" in action_type:
            u_perf = 0.85 if high_load else 0.5
        else:
            u_perf = 0.5

        # Cascade backpressure adjustment: If downstream dependencies are stressed,
        # penalize capacity-expanding actions and prioritize load-shedding/throttling.
        if downstream_stressed:
            is_load_shedding = (
                "dimmer" in action_type
                or "throttle" in action_type
                or (
                    schema is not None
                    and schema.cost_impact == "positive"
                    and schema.performance_impact != "positive"
                )
            )
            is_capacity_expansion = "scale_up" in action_type or (
                schema is not None and schema.performance_impact == "positive"
            )
            if is_capacity_expansion:
                u_perf = 0.2
            elif is_load_shedding:
                u_perf = 0.95

        # 2. Cost utility (higher score = lower monetary / server footprint)
        if schema and getattr(schema, "cost_impact", "neutral") != "neutral":
            if schema.cost_impact == "positive":
                u_cost = 1.0
            else:
                u_cost = 0.3
        elif "scale_down" in action_type:
            u_cost = 1.0
        elif "dimmer" in action_type:
            u_cost = 0.85
        elif "scale_up" in action_type:
            u_cost = 0.3
        else:
            u_cost = 0.7

        # 3. QoS utility (preservation of content quality)
        if schema and getattr(schema, "qos_impact", "neutral") != "neutral":
            if schema.qos_impact == "positive":
                u_qos = 1.0
            else:
                u_qos = 0.5 if high_load else 0.8
        elif "scale_up" in action_type:
            u_qos = 1.0
        elif "dimmer" in action_type:
            dimmer_val = 1.0
            if action.parameters and "dimmer" in action.parameters:
                try:
                    dimmer_val = float(action.parameters["dimmer"])
                except (TypeError, ValueError):
                    pass
            u_qos = max(0.2, min(1.0, dimmer_val))
        elif "scale_down" in action_type:
            u_qos = 0.5 if high_load else 0.8
        else:
            u_qos = 0.8

        w_perf = self.objective_weights.get("performance", 0.5)
        w_cost = self.objective_weights.get("cost", 0.3)
        w_qos = self.objective_weights.get("qos", 0.2)

        total_weight = w_perf + w_cost + w_qos
        if total_weight <= 0:
            total_weight = 1.0

        composite = ((w_perf * u_perf) + (w_cost * u_cost) + (w_qos * u_qos)) / total_weight

        # Weight composite utility with confidence
        return composite * max(0.1, min(1.0, confidence))

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
            allowed_values=["first", "priority", "confidence", "pareto"],
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
            if len(parts) != 2:
                return False
            try:
                strategy_idx = int(parts[0].split("_")[1])
            except (IndexError, ValueError):
                return False
            sub_path = parts[1]

            if strategy_idx < len(self.strategies):
                return await self.strategies[strategy_idx][0].update_parameter(sub_path, new_value)

        elif parameter_path == "selection_mode":
            if new_value in ["first", "priority", "confidence", "pareto", "multi_objective"]:
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
            for sub_conf, (sub_strategy, _priority) in zip(new_subs, self.strategies):
                if not isinstance(sub_conf, dict):
                    continue

                sub_params = sub_conf.get("params", {})
                if sub_params is None:
                    sub_params = {}
                if not isinstance(sub_params, dict):
                    continue

                await sub_strategy.apply_config_update(sub_params)

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
