"""Statistical world model implementation."""

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.abstractions.observability import Logger, MetricsCollector
from polaris.abstractions.world_model import PredictionResult, WorldModel
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

    async def predict(
        self, action: AdaptationAction, current_state: SystemState
    ) -> PredictionResult:
        """Predict outcome of action.

        Simple prediction: use historical mean as baseline.
        """
        if self._metrics:
            self._metrics.increment(
                "polaris.world_model.statistical.predictions",
                tags={"system_id": current_state.system_id},
            )

        predicted = {}
        system_id = action.target_system
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

        if self._use_kalman and confidences:
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.5

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

        return PredictionResult(
            predicted_metrics=predicted,
            confidence=confidence,
            reasoning=reasoning,
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
