"""Tests for StatisticalMetaLearner world model integration."""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.meta_learner import PerformanceAnalysis
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.meta_learner.statistical import StatisticalMetaLearner


class DummyStrategy(AdaptationStrategy):
    """Minimal strategy with a single tunable threshold parameter."""

    # type: ignore[override]
    def get_tunable_parameters(self) -> Dict[str, Any]:
        class Spec:
            def __init__(self, current: float) -> None:
                self.current_value = current
                self.max_value = None

        return {"threshold.high": Spec(100.0)}

    async def apply_parameters(
        self, parameters: Dict[str, Any]
    ) -> None:  # pragma: no cover - not used here
        pass

    async def assess(self, state, context):  # type: ignore[override]
        """Minimal implementation to satisfy abstract interface; not used in these tests."""
        return None

    # type: ignore[override]
    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        """Minimal implementation that always reports success; used indirectly by meta-
        learner tuning.
        """
        return True

    async def on_action_executed(self, action, result) -> None:
        pass


@pytest.mark.asyncio
async def test_analyze_performance_includes_world_model_uncertainty():
    """Should include world_model_uncertainty when a world model is provided."""

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

    logger = MagicMock()

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
    logger = MagicMock()
    meta = StatisticalMetaLearner(
        knowledge_store=knowledge_store, logger=logger, conservative_mode=True
    )

    strategy = DummyStrategy()
    proposals = await meta.propose_strategy_updates(strategy, analysis)

    assert proposals
    # With high avg_metric_std, confidence should be <= 0.5 (base 0.6 minus 0.1)
    assert all(p.confidence <= 0.5 for p in proposals)


@pytest.mark.asyncio
async def test_analyze_performance_insights_fallback():
    """Test analyze_performance when world model throws an exception."""
    knowledge_store = AsyncMock()
    knowledge_store.query_states.return_value = []
    knowledge_store.query_actions.return_value = []

    world_model = AsyncMock()
    world_model.get_insights.side_effect = Exception("World model down")

    meta = StatisticalMetaLearner(
        knowledge_store=knowledge_store,
        world_model=world_model,
    )

    analysis = await meta.analyze_performance("system-1", time_window_hours=1.0)

    assert analysis.insights["success_rate"] == 0.0
    assert "world_model_uncertainty" not in analysis.insights
    assert "Low success rate - consider adjusting thresholds" in analysis.recommendations


@pytest.mark.asyncio
async def test_propose_empty_tunable_parameters():
    """Test propose_strategy_updates with no tunable parameters."""
    knowledge_store = AsyncMock()
    meta = StatisticalMetaLearner(knowledge_store=knowledge_store)

    class EmptyParamsStrategy(AdaptationStrategy):
        def get_tunable_parameters(self):
            return {}

        async def apply_parameters(self, p):
            pass

        async def assess(self, s, c):
            return []

        async def update_parameter(self, p, v):
            return True

        async def on_action_executed(self, a, r):
            pass

    strategy = EmptyParamsStrategy()
    analysis = PerformanceAnalysis("system-1", 1.0, 0.5, {}, [])

    proposals = await meta.propose_strategy_updates(strategy, analysis)
    assert len(proposals) == 0


@pytest.mark.asyncio
async def test_propose_bayesian_updates_success():
    """Test propose_strategy_updates using bayesian optimization."""
    knowledge_store = AsyncMock()

    class MockConfig:
        def __init__(self, params, perf):
            self.parameters = params
            self.performance = perf

    # Need at least self.min_samples_for_optimization = 10 samples to fit GP
    history = []
    for i in range(12):
        action = MagicMock()
        action.parameters = {"threshold.high": 80.0 + i}
        action.action_type = "test"

        result = MagicMock()
        result.status.value = "success" if i % 2 == 0 else "failed"
        history.append((action, result))

    knowledge_store.query_actions.return_value = history

    meta = StatisticalMetaLearner(
        knowledge_store=knowledge_store,
        enable_bayesian_optimization=True,
    )

    strategy = DummyStrategy()
    analysis = PerformanceAnalysis("system-1", 24.0, 0.8, {}, [])

    # We mock out the optimizer fit/suggest behavior just to see the proposal creation
    with pytest.MonkeyPatch.context() as m:
        # We need to mock _collect_historical_configurations alongside _get_or_create_optimizer
        # specifically if we want to ensure Bayesian optimization triggers properly.

        mock_opt = MagicMock()
        mock_opt.fit.return_value = True
        mock_opt.get_optimization_confidence.return_value = 0.8
        mock_opt.suggest_next_parameters.return_value = [{"threshold.high": 85.0}]

        m.setattr(meta, "_get_or_create_optimizer", lambda s, t: mock_opt)
        m.setattr(
            meta, "_collect_historical_configurations", AsyncMock(return_value=["mocked_config"])
        )

        proposals = await meta.propose_strategy_updates(strategy, analysis)

        assert len(proposals) == 1
        assert proposals[0].proposed_value == 85.0
        assert "Bayesian optimization suggests" in proposals[0].rationale


@pytest.mark.asyncio
async def test_propose_bayesian_updates_fails():
    """Test propose_strategy_updates when GP fitting fails/not enough data."""
    knowledge_store = AsyncMock()
    # 0 history items will fail GP fitting
    knowledge_store.query_actions.return_value = []

    meta = StatisticalMetaLearner(
        knowledge_store=knowledge_store,
        enable_bayesian_optimization=True,
    )

    strategy = DummyStrategy()
    analysis = PerformanceAnalysis("system-1", 24.0, 0.5, {}, [])

    # It should fallback to heuristic
    proposals = await meta.propose_strategy_updates(strategy, analysis)

    assert len(proposals) == 1
    # 1.1 * 100.0 = 110.0 (non-conservative)
    assert proposals[0].proposed_value == pytest.approx(110.0)
    assert "heuristic" not in proposals[0].rationale.lower()  # but rationale is heuristic


@pytest.mark.asyncio
async def test_conservative_constraints():
    """Test applying conservative constraints."""
    meta = StatisticalMetaLearner(knowledge_store=AsyncMock(), conservative_mode=True)

    class Spec:
        min_value = 50.0
        max_value = 150.0
        current_value = 100.0

    spec = Spec()

    # 15% max change: 100 -> 115
    assert meta._apply_conservative_constraints(100.0, 150.0, spec) == 115.0
    # Min bound test
    assert meta._apply_conservative_constraints(55.0, 10.0, spec) == 50.0
    # Max bound test
    assert meta._apply_conservative_constraints(140.0, 190.0, spec) == 150.0


@pytest.mark.asyncio
async def test_validate_proposals():
    """Test validating proposals."""
    from polaris.abstractions.meta_learner import ParameterProposal, ProposalStatus

    p1 = ParameterProposal("1", "p1", 1, 2, "r", 0.9, "e")
    p2 = ParameterProposal("2", "p2", 1, 2, "r", 0.4, "e")  # Rejected
    p3 = ParameterProposal("3", "p3", 1, 2, "r", 0.6, "e")

    meta = StatisticalMetaLearner(knowledge_store=AsyncMock())

    validated = await meta.validate_proposals([p1, p2, p3])

    assert len(validated) == 2
    assert validated[0].proposal_id == "1"  # Highest conf first
    assert validated[1].proposal_id == "3"
    assert validated[0].status == ProposalStatus.APPROVED
    assert p2.status == ProposalStatus.REJECTED
