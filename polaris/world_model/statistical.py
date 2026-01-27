"""
Statistical world model implementation.
"""

from typing import Dict, Any
from collections import defaultdict
import statistics

from polaris.abstractions.world_model import WorldModel, PredictionResult
from polaris.abstractions.knowledge_store import KnowledgeStore
from polaris.core.models import SystemState, AdaptationAction


class StatisticalWorldModel(WorldModel):
    """
    Statistical world model using mean/std calculations.

    Tracks metric trends and provides simple predictions.
    """

    def __init__(self, knowledge_store: KnowledgeStore):
        self.knowledge_store = knowledge_store
        self._metric_history: Dict[str, Dict[str, list]
                                   ] = defaultdict(lambda: defaultdict(list))

    async def update(self, state: SystemState) -> None:
        """Update model with new state."""
        for metric_name, metric in state.metrics.items():
            try:
                value = float(metric.value)
                self._metric_history[state.system_id][metric_name].append(
                    value)

                # Keep only last 100 values
                if len(self._metric_history[state.system_id][metric_name]) > 100:
                    self._metric_history[state.system_id][metric_name] = \
                        self._metric_history[state.system_id][metric_name][-100:]
            except (ValueError, TypeError):
                continue

    async def predict(
        self,
        action: AdaptationAction,
        current_state: SystemState
    ) -> PredictionResult:
        """
        Predict outcome of action.

        Simple prediction: use historical mean as baseline.
        """
        predicted = {}
        system_id = action.target_system

        for metric_name in current_state.metrics.keys():
            history = self._metric_history[system_id].get(metric_name, [])
            if history:
                predicted[metric_name] = statistics.mean(history)

        return PredictionResult(
            predicted_metrics=predicted,
            confidence=0.5,  # Low confidence for statistical model
            reasoning="Statistical baseline from historical mean"
        )

    async def get_insights(self) -> Dict[str, Any]:
        """Get simple statistical insights."""
        insights = {}
        for system_id, metrics in self._metric_history.items():
            insights[system_id] = {}
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    insights[system_id][metric_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values)
                    }
        return insights
