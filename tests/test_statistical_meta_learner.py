"""Tests for StatisticalMetaLearner world model integration."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from polaris.meta_learner.statistical import StatisticalMetaLearner
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.abstractions.meta_learner import PerformanceAnalysis


class DummyStrategy(AdaptationStrategy):
    """Minimal strategy with a single tunable threshold parameter."""

    def get_tunable_parameters(self) -> Dict[str, Any]:  # type: ignore[override]
        class Spec:
            def __init__(self, current: float) -> None:
                self.current_value = current
                self.max_value = None

        return {"threshold.high": Spec(100.0)}

    async def apply_parameters(self, parameters: Dict[str, Any]) -> None:  # pragma: no cover - not used here
        pass

    async def assess(self, state, context):  # type: ignore[override]
        """Minimal implementation to satisfy abstract interface; not used in these tests."""
        return None

    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:  # type: ignore[override]
        """Minimal implementation that always reports success; used indirectly by meta-learner tuning."""
        return True


@pytest.mark.asyncio
async def test_analyze_performance_includes_world_model_uncertainty():
    """StatisticalMetaLearner should include world_model_uncertainty when a world model is provided."""

    # Mock knowledge store with no historical states/actions
    knowledge_store = AsyncMock()
    knowledge_store.query_states.return_value = []
    knowledge_store.query_actions.return_value = []

    # World model get_insights returns stds and a regime for the system
    world_model = AsyncMock()
    world_model.get_insights.return_value = {
        "system-1": {
            "metric_a": {"std": 5.0},
            "metric_b": {"std": 15.0},
            "regime": {
                "probabilities": {"low": 0.1, "normal": 0.2, "high": 0.7},
                "most_likely": "high",
            },
        }
    }

    logger = AsyncMock()

    meta = StatisticalMetaLearner(
        knowledge_store=knowledge_store,
        logger=logger,
        conservative_mode=True,
        world_model=world_model,
    )

    analysis = await meta.analyze_performance("system-1", time_window_hours=1.0)

    assert "world_model_uncertainty" in analysis.insights
    wm_unc = analysis.insights["world_model_uncertainty"]
    assert wm_unc["avg_metric_std"] > 0.0
    assert "regime" in wm_unc


@pytest.mark.asyncio
async def test_proposal_confidence_adjusted_by_world_model_uncertainty():
    """Proposal confidence should be reduced when world model uncertainty is high."""

    # Prepare a PerformanceAnalysis with embedded world_model_uncertainty
    insights = {
        "world_model_uncertainty": {"avg_metric_std": 20.0},
        "total_adaptations": 10,
    }
    analysis = PerformanceAnalysis(
        system_id="system-1",
        time_window_hours=1.0,
        success_rate=0.5,  # low success rate to trigger proposals
        insights=insights,
        recommendations=[],
    )

    knowledge_store = AsyncMock()
    logger = AsyncMock()
    meta = StatisticalMetaLearner(knowledge_store=knowledge_store, logger=logger, conservative_mode=True)

    strategy = DummyStrategy()
    proposals = await meta.propose_strategy_updates(strategy, analysis)

    assert proposals
    # With high avg_metric_std, confidence should be <= 0.5 (base 0.6 minus 0.1)
    assert all(p.confidence <= 0.5 for p in proposals)
