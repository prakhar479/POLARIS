"""Statistical world model implementation."""

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.world_model import DomainSurrogate, PredictionResult, WorldModel
from polaris.core.models import AdaptationAction, SystemState


class _ScalarKalmanFilter:
    """A simple scalar Kalman filter for smoothing noisy metric values."""

    def __init__(self, process_var: float = 1.0, measurement_var: float = 1.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self._x: Optional[float] = None
        self._p: Optional[float] = None

    def update(self, z: float) -> None:
        if self._x is None or self._p is None:
            self._x = z
            self._p = self.process_var
            return

        p_prior = self._p + self.process_var
        k = p_prior / (p_prior + self.measurement_var)
        self._x = self._x + k * (z - self._x)
        self._p = (1.0 - k) * p_prior

    def predict(self) -> Optional[tuple]:
        if self._x is None or self._p is None:
            return None
        p_prior = self._p + self.process_var
        return self._x, p_prior


class StatisticalWorldModel(WorldModel):
    """Statistical world model using mean/std calculations.

    Tracks metric trends and provides simple predictions.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        use_kalman: bool = False,
        window_size: int = 100,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        """Initialize the statistical world model.

        Args:
            knowledge_store: Knowledge store for retrieving historical data
            use_kalman: Whether to use Kalman filtering for predictions
            window_size: Number of recent metric values retained per system/metric
            logger: Logger for logging events
            metrics: Metrics collector for tracking performance
        """
        self.knowledge_store = knowledge_store
        self._window_size = max(1, int(window_size))
        self._metric_history: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._use_kalman = use_kalman
        self._kalman_filters: Dict[str, Dict[str, _ScalarKalmanFilter]] = defaultdict(dict)
        # Simple HMM-style regime tracking per system
        self._regimes: List[str] = ["low", "normal", "high"]
        self._regime_probs: Dict[str, Dict[str, float]] = {}
        # Empirical action effects: system_id -> action_type -> metric_name -> list of observed deltas
        self._action_effects: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # Pending in-flight actions awaiting post-adaptation observation
        self._pending_actions: Dict[str, Tuple[AdaptationAction, SystemState]] = {}
        # Pluggable domain physics/queuing surrogates
        from polaris.world_model.surrogates import (
            ModelSwitchingDomainSurrogate,
            QueuingDomainSurrogate,
        )

        self._surrogates: List[DomainSurrogate] = [
            QueuingDomainSurrogate(),
            ModelSwitchingDomainSurrogate(),
        ]
        self._logger = logger
        self._metrics = metrics

        if self._logger:
            self._logger.info(
                "StatisticalWorldModel initialized",
                use_kalman=self._use_kalman,
                window_size=self._window_size,
            )

        if self._metrics:
            self._metrics.increment("polaris.world_model.statistical.initialized")

    async def update(self, state: SystemState) -> None:
        """Update model with new state."""
        if self._metrics:
            self._metrics.increment(
                "polaris.world_model.statistical.updates",
                tags={"system_id": state.system_id},
            )

        # Absorb observed action effect if an action was pending on this system
        pending = self._pending_actions.pop(state.system_id, None)
        if pending is not None:
            pending_action, pre_state = pending
            observed_deltas: Dict[str, float] = {}
            for m_name, m_curr in state.metrics.items():
                if m_name in pre_state.metrics:
                    try:
                        curr_v = float(m_curr.value)
                        prev_v = float(pre_state.metrics[m_name].value)
                        observed_deltas[m_name] = round(curr_v - prev_v, 4)
                    except (ValueError, TypeError):
                        pass
            if observed_deltas:
                self.record_action_effect(
                    state.system_id, pending_action.action_type, observed_deltas
                )

        values_recorded = 0
        for metric_name, metric in state.metrics.items():
            try:
                value = float(metric.value)
                self._metric_history[state.system_id][metric_name].append(value)

                # Keep only recent values for memory-bounded insights/prediction.
                if len(self._metric_history[state.system_id][metric_name]) > self._window_size:
                    self._metric_history[state.system_id][metric_name] = self._metric_history[
                        state.system_id
                    ][metric_name][-self._window_size :]

                values_recorded += 1

                if self._use_kalman:
                    system_filters = self._kalman_filters[state.system_id]
                    filt = system_filters.get(metric_name)
                    if filt is None:
                        filt = _ScalarKalmanFilter()
                        system_filters[metric_name] = filt
                    filt.update(value)
            except (ValueError, TypeError) as e:
                if self._logger:
                    self._logger.warning(
                        "Failed to parse world model metric value",
                        system_id=state.system_id,
                        metric=metric_name,
                        raw_value=getattr(metric, "value", None),
                        error=str(e),
                    )
                if self._metrics:
                    self._metrics.increment(
                        "polaris.world_model.statistical.parse_errors",
                        tags={"system_id": state.system_id, "metric": metric_name},
                    )
                continue

        if self._metrics and values_recorded:
            self._metrics.increment(
                "polaris.world_model.statistical.values_recorded",
                value=values_recorded,
            )

        # Update simple regime probabilities using a heuristic on key metrics
        self._update_regime(state)

    def _update_regime(self, state: SystemState) -> None:
        """Update HMM-style regime probabilities based on current metrics.

        Uses a simple Hidden Markov Model approach to track system operating
        regimes (low, normal, high load). Updates transition probabilities
        using emission preferences derived from CPU usage and response time.

        Args:
            state: Current system state containing metrics to analyze.
        """
        system_id = state.system_id
        if system_id not in self._regime_probs:
            # Start with uniform prior over regimes
            self._regime_probs[system_id] = {
                name: 1.0 / len(self._regimes) for name in self._regimes
            }

        probs = self._regime_probs[system_id]

        cpu = state.metrics.get("cpu_usage")
        resp = state.metrics.get("response_time")
        try:
            cpu_val = float(cpu.value) if cpu is not None else None
        except (ValueError, TypeError):
            cpu_val = None
        try:
            resp_val = float(resp.value) if resp is not None else None
        except (ValueError, TypeError):
            resp_val = None

        # Simple emission preferences based on CPU / response time levels
        emission: Dict[str, float] = dict.fromkeys(self._regimes, 1.0)
        if cpu_val is not None:
            if cpu_val < 40.0:
                emission["low"] *= 2.0
            elif cpu_val > 80.0:
                emission["high"] *= 2.0
            else:
                emission["normal"] *= 2.0
        if resp_val is not None:
            if resp_val > 500.0:
                emission["high"] *= 1.5
            elif resp_val < 200.0:
                emission["low"] *= 1.2

        # Fixed self-biased transition (Markovian smoothing)
        stay_bias = 0.7
        move_bias = (1.0 - stay_bias) / max(len(self._regimes) - 1, 1)
        new_probs: Dict[str, float] = {}
        for target in self._regimes:
            prior = 0.0
            for source in self._regimes:
                if source == target:
                    prior += stay_bias * probs[source]
                else:
                    prior += move_bias * probs[source]
            new_probs[target] = prior * emission[target]

        # Normalize
        total = sum(new_probs.values())
        if total > 0.0:
            for name in self._regimes:
                new_probs[name] = new_probs[name] / total
            self._regime_probs[system_id] = new_probs

    def register_surrogate(self, surrogate: DomainSurrogate) -> None:
        """Register a custom domain physics, queuing, or surrogate model."""
        self._surrogates.append(surrogate)

    def record_pending_action(self, action: AdaptationAction, pre_state: SystemState) -> None:
        """Register an in-flight action to automatically record deltas on next state update."""
        system_id = action.target_system or pre_state.system_id
        self._pending_actions[system_id] = (action, pre_state)

    def record_action_effect(
        self, system_id: str, action_type: str, metric_deltas: Dict[str, float]
    ) -> None:
        """Record observed metric deltas resulting from an executed action."""
        for metric_name, delta in metric_deltas.items():
            try:
                val = float(delta)
                history = self._action_effects[system_id][action_type][metric_name]
                history.append(val)
                if len(history) > self._window_size:
                    self._action_effects[system_id][action_type][metric_name] = history[
                        -self._window_size :
                    ]
            except (ValueError, TypeError):
                continue

        if self._metrics:
            self._metrics.increment(
                "polaris.world_model.statistical.action_effects_recorded",
                tags={"system_id": system_id, "action_type": action_type},
            )

    async def predict(
        self, action: AdaptationAction, current_state: SystemState
    ) -> PredictionResult:
        """Predict outcome of action using baseline statistics and action-effect dynamics."""
        if self._metrics:
            self._metrics.increment(
                "polaris.world_model.statistical.predictions",
                tags={"system_id": current_state.system_id},
            )

        predicted = {}
        system_id = action.target_system or current_state.system_id
        confidences: List[float] = []

        for metric_name in current_state.metrics.keys():
            history = self._metric_history[system_id].get(metric_name, [])
            if not history:
                continue
            if self._use_kalman:
                system_filters = self._kalman_filters.get(system_id, {})
                filt = system_filters.get(metric_name)
                if filt is not None:
                    prediction = filt.predict()
                    if prediction is not None:
                        mean, variance = prediction
                        predicted[metric_name] = mean
                        conf = 1.0 / (1.0 + max(variance, 0.0))
                        confidences.append(conf)
                        continue
            predicted[metric_name] = statistics.mean(history)

        if not predicted:
            return PredictionResult(
                predicted_metrics={},
                confidence=0.5,
                reasoning="Statistical baseline from historical mean",
                uncertainty={},
            )

        # Check for empirical action effects
        applied_deltas: Dict[str, float] = {}
        effects_for_action = self._action_effects.get(system_id, {}).get(action.action_type, {})
        for metric_name, deltas in effects_for_action.items():
            if deltas and metric_name in predicted:
                avg_delta = statistics.mean(deltas)
                predicted[metric_name] = max(0.0, predicted[metric_name] + avg_delta)
                applied_deltas[metric_name] = avg_delta

        # If no empirical effects recorded yet, apply pluggable domain surrogates
        if not effects_for_action:
            for surrogate in self._surrogates:
                if surrogate.can_handle(system_id, action.action_type):
                    surr_deltas = surrogate.predict_deltas(action, current_state)
                    for m_name, d_val in surr_deltas.items():
                        if m_name in predicted:
                            predicted[m_name] = round(max(0.0, predicted[m_name] + d_val), 4)
                            applied_deltas[m_name] = d_val
                    break

        # Check for direct parameter assignments (e.g. set_dimmer)
        if action.parameters and isinstance(action.parameters, dict):
            for p_key, p_val in action.parameters.items():
                if p_key in current_state.metrics:
                    try:
                        p_float = float(p_val)
                        predicted[p_key] = p_float
                        curr_mv = current_state.metrics[p_key]
                        curr_val = float(curr_mv.value) if curr_mv is not None else 0.0
                        applied_deltas[p_key] = p_float - curr_val
                    except (TypeError, ValueError):
                        pass

        if self._use_kalman and confidences:
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.5

        if applied_deltas:
            confidence = min(0.95, max(confidence, 0.75))

        if self._metrics:
            self._metrics.histogram(
                "polaris.world_model.statistical.prediction_confidence",
                confidence,
                tags={"system_id": system_id},
            )

        reasoning_parts: List[str] = []
        if self._use_kalman:
            reasoning_parts.append(
                "Kalman-smoothed statistical prediction with variance-based confidence"
            )
        else:
            reasoning_parts.append("Statistical baseline from historical mean")

        if applied_deltas:
            delta_str = ", ".join(f"{m}: {d:+.2f}" for m, d in applied_deltas.items())
            reasoning_parts.append(
                f"Simulated counterfactual impact for action '{action.action_type}' ({delta_str})"
            )

        # Add regime information if available
        regime_info = self._regime_probs.get(system_id)
        if regime_info:
            most_likely = max(regime_info.items(), key=lambda x: x[1])
            reasoning_parts.append(f"Estimated regime: {most_likely[0]} (p={most_likely[1]: .2f})")

        reasoning = "; ".join(reasoning_parts)

        if self._logger:
            self._logger.debug(
                "World model prediction generated",
                system_id=system_id,
                metric_count=len(predicted),
                confidence=confidence,
            )

        # Compute metric uncertainties
        uncertainties: Dict[str, float] = {}
        for metric_name, pred_val in predicted.items():
            if self._use_kalman and metric_name in self._kalman_filters.get(system_id, {}):
                filt = self._kalman_filters[system_id][metric_name]
                p_out = filt.predict()
                uncertainties[metric_name] = round(p_out[1], 4) if p_out is not None else 1.0
            else:
                hist = self._metric_history[system_id].get(metric_name, [])
                if len(hist) > 1:
                    uncertainties[metric_name] = round(statistics.variance(hist), 4)
                else:
                    uncertainties[metric_name] = round(0.05 * max(abs(pred_val), 1.0), 4)

        return PredictionResult(
            predicted_metrics=predicted,
            confidence=confidence,
            reasoning=reasoning,
            uncertainty=uncertainties,
        )

    async def get_insights(self) -> Dict[str, Any]:
        """Get simple statistical insights."""
        if self._metrics:
            self._metrics.increment("polaris.world_model.statistical.insights_requested")

        insights: Dict[str, Dict[str, Any]] = {}
        for system_id, metrics in self._metric_history.items():
            insights[system_id] = {}
            has_metric_insights = False
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    insights[system_id][metric_name] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                    }
                    has_metric_insights = True

            # Attach regime information only if we have at least one metric insight
            regime_probs = self._regime_probs.get(system_id)
            if has_metric_insights and regime_probs:
                most_likely = max(regime_probs.items(), key=lambda x: x[1])
                insights[system_id]["regime"] = {
                    "probabilities": regime_probs,
                    "most_likely": most_likely[0],
                }
        if self._metrics:
            self._metrics.gauge(
                "polaris.world_model.statistical.systems_with_insights",
                len(insights),
            )
        return insights

    def is_stressed(self, system_id: str) -> bool:
        """Check if system is in high regime based on statistical regime probabilities."""
        regime_probs = self._regime_probs.get(system_id, {})
        return bool(regime_probs.get("high", 0.0) > 0.5)
