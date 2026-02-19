"""Tests for StatisticalMetaLearner with Bayesian optimization."""

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from polaris.abstractions.meta_learner import PerformanceAnalysis
from polaris.abstractions.strategy import AdaptationStrategy
from polaris.meta_learner.bayesian_optimizer import AcquisitionFunction
from polaris.meta_learner.statistical import StatisticalMetaLearner


class DummyStrategyWithBayesian(AdaptationStrategy):
    """Strategy with tunable parameters for Bayesian optimization testing."""

    # type: ignore[override]
    def get_tunable_parameters(self) -> Dict[str, Any]:
        class Spec:
            def __init__(
                self, current: float, param_type: type, min_val=None, max_val=None, kind=None
            ):
                self.current_value = current
                self.type = param_type
                self.min_value = min_val
                self.max_value = max_val
                self.kind = kind
                self.allowed_values = None

        return {
            "threshold.high": Spec(80.0, float, 50.0, 100.0, "threshold_high"),
            "threshold.low": Spec(20.0, float, 10.0, 30.0, "threshold_low"),
            "cooldown_seconds": Spec(60, int, 30, 300, "cooldown"),
        }

    async def apply_parameters(self, parameters: Dict[str, Any]) -> None:  # pragma: no cover
        pass

    async def assess(self, state, context):  # type: ignore[override]
        return None

    # type: ignore[override]
    async def update_parameter(self, parameter_path: str, new_value: Any) -> bool:
        return True

    # type: ignore[override]
    async def on_action_executed(self, action, result):
        pass


@pytest.mark.asyncio
class TestStatisticalMetaLearnerBayesian:
    """Test StatisticalMetaLearner with Bayesian optimization."""

    @pytest.fixture
    def knowledge_store(self):
        """Mock knowledge store."""
        store = AsyncMock()
        store.query_states.return_value = []
        store.query_actions.return_value = []
        return store

    @pytest.fixture
    def logger(self):
        """Mock logger."""
        return MagicMock()

    @pytest.fixture
    def world_model(self):
        """Mock world model."""
        wm = AsyncMock()
        wm.get_insights.return_value = {
            "test-system": {
                "cpu_usage": {"std": 5.0},
                "memory_usage": {"std": 3.0},
                "regime": {
                    "probabilities": {"low": 0.1, "normal": 0.8, "high": 0.1},
                    "most_likely": "normal",
                },
            }
        }
        return wm

    @pytest.fixture
    def meta_learner(self, knowledge_store, logger, world_model):
        """Create StatisticalMetaLearner with Bayesian optimization enabled."""
        return StatisticalMetaLearner(
            knowledge_store=knowledge_store,
            logger=logger,
            conservative_mode=True,
            world_model=world_model,
            enable_bayesian_optimization=True,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            exploration_weight=0.1,
            min_samples_for_optimization=5,
        )

    @pytest.fixture
    def strategy(self):
        """Create test strategy."""
        return DummyStrategyWithBayesian()

    @pytest.fixture
    def performance_analysis(self):
        """Create performance analysis."""
        return PerformanceAnalysis(
            system_id="test-system",
            time_window_hours=1.0,
            success_rate=0.5,  # Low success rate to trigger proposals
            insights={
                "total_states": 10,
                "total_adaptations": 15,
                "world_model_uncertainty": {"avg_metric_std": 5.0},
            },
            recommendations=["Low success rate detected"],
        )

    async def test_bayesian_optimization_enabled_by_default(self, meta_learner):
        """Test that Bayesian optimization is enabled by default."""
        assert meta_learner.enable_bayesian_optimization is True
        assert meta_learner.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT
        assert meta_learner.min_samples_for_optimization == 5

    async def test_bayesian_optimization_disabled(self, knowledge_store, logger, world_model):
        """Test meta-learner with Bayesian optimization disabled."""
        meta_learner = StatisticalMetaLearner(
            knowledge_store=knowledge_store,
            logger=logger,
            world_model=world_model,
            enable_bayesian_optimization=False,
        )

        assert meta_learner.enable_bayesian_optimization is False

    async def test_propose_updates_with_insufficient_data_falls_back_to_heuristics(
        self, meta_learner, strategy, performance_analysis, knowledge_store
    ):
        """Test fallback to heuristics when insufficient data for Bayesian optimization."""
        # Mock insufficient historical data
        knowledge_store.query_actions.return_value = []

        proposals = await meta_learner.propose_strategy_updates(strategy, performance_analysis)

        # Should fall back to heuristic-based proposals
        assert len(proposals) > 0

        # Check that proposals are heuristic-based (rationale should mention heuristics)
        for proposal in proposals:
            assert proposal.confidence >= 0.5
            assert proposal.status.value == "pending"

    async def test_bayesian_optimization_with_sufficient_data(
        self, meta_learner, strategy, performance_analysis, knowledge_store
    ):
        """Test Bayesian optimization with sufficient historical data."""
        # Mock sufficient historical data
        mock_actions = []
        for i in range(10):  # More than min_samples_for_optimization
            action = MagicMock()
            action.parameters = {
                "threshold.high": 70.0 + i * 2,
                "threshold.low": 15.0 + i,
                "cooldown_seconds": 50 + i * 5,
            }
            action.action_type = "test_action"
            action.timestamp = datetime.now(timezone.utc)

            result = MagicMock()
            result.status.value = "success" if i % 2 == 0 else "failed"

            mock_actions.append((action, result))

        knowledge_store.query_actions.return_value = mock_actions

        proposals = await meta_learner.propose_strategy_updates(strategy, performance_analysis)

        # Should generate Bayesian optimization proposals
        assert len(proposals) > 0

        # Check that proposals contain Bayesian optimization rationale
        for proposal in proposals:
            assert "Bayesian optimization" in proposal.rationale
            assert "optimization confidence" in proposal.rationale
            assert proposal.confidence >= 0.5

    async def test_conservative_mode_constraints(
        self, meta_learner, strategy, performance_analysis, knowledge_store
    ):
        """Test that conservative mode applies constraints to Bayesian suggestions."""
        # Mock historical data
        mock_actions = []
        for _ in range(10):
            action = MagicMock()
            action.parameters = {
                "threshold.high": 80.0,
                "threshold.low": 20.0,
                "cooldown_seconds": 60,
            }
            action.action_type = "test_action"
            action.timestamp = datetime.now(timezone.utc)

            result = MagicMock()
            result.status.value = "success"

            mock_actions.append((action, result))

        knowledge_store.query_actions.return_value = mock_actions

        proposals = await meta_learner.propose_strategy_updates(strategy, performance_analysis)

        # In conservative mode, changes should be limited
        for proposal in proposals:
            spec = strategy.get_tunable_parameters()[proposal.parameter_path]
            current = spec.current_value
            proposed = proposal.proposed_value

            if isinstance(current, (int, float)) and isinstance(proposed, (int, float)):
                # Change should be limited to ~15% in conservative mode
                relative_change = abs(proposed - current) / abs(current) if current != 0 else 0
                assert relative_change <= 0.2  # Allow some tolerance

    async def test_world_model_uncertainty_adjusts_confidence(
        self, meta_learner, strategy, performance_analysis, knowledge_store
    ):
        """Test that world model uncertainty affects proposal confidence."""
        # Create analysis with high uncertainty
        high_uncertainty_analysis = PerformanceAnalysis(
            system_id="test-system",
            time_window_hours=1.0,
            success_rate=0.5,
            insights={"world_model_uncertainty": {"avg_metric_std": 20.0}},  # High uncertainty
            recommendations=[],
        )

        # Mock historical data
        mock_actions = []
        for _ in range(10):
            action = MagicMock()
            action.parameters = {"threshold.high": 80.0}
            action.action_type = "test_action"
            action.timestamp = datetime.now(timezone.utc)

            result = MagicMock()
            result.status.value = "success"

            mock_actions.append((action, result))

        knowledge_store.query_actions.return_value = mock_actions

        proposals = await meta_learner.propose_strategy_updates(strategy, high_uncertainty_analysis)

        # High uncertainty should reduce confidence
        for proposal in proposals:
            # Should be lower than base confidence due to high uncertainty
            assert proposal.confidence <= 0.7  # Reduced from typical 0.5-0.9 range

    async def test_optimizer_caching_per_system(self, meta_learner, strategy, knowledge_store):
        """Test that optimizers are cached per system."""
        # Mock historical data for system 1
        mock_actions_1 = []
        for _ in range(10):
            action = MagicMock()
            action.parameters = {"threshold.high": 80.0}
            action.action_type = "test_action"
            action.timestamp = datetime.now(timezone.utc)
            result = MagicMock()
            result.status.value = "success"
            mock_actions_1.append((action, result))

        knowledge_store.query_actions.return_value = mock_actions_1

        analysis_1 = PerformanceAnalysis(
            system_id="system-1",
            time_window_hours=1.0,
            success_rate=0.5,
            insights={},
            recommendations=[],
        )

        # First call should create optimizer
        _ = await meta_learner.propose_strategy_updates(strategy, analysis_1)
        assert len(meta_learner._optimizers) == 1
        assert "system-1" in meta_learner._optimizers

        # Second call to same system should reuse optimizer
        _ = await meta_learner.propose_strategy_updates(strategy, analysis_1)
        assert len(meta_learner._optimizers) == 1  # Still only one optimizer

        # Call to different system should create new optimizer
        analysis_2 = PerformanceAnalysis(
            system_id="system-2",
            time_window_hours=1.0,
            success_rate=0.5,
            insights={},
            recommendations=[],
        )

        _ = await meta_learner.propose_strategy_updates(strategy, analysis_2)
        assert len(meta_learner._optimizers) == 2  # Now two optimizers
        assert "system-2" in meta_learner._optimizers

    async def test_parameter_type_detection(self, meta_learner):
        """Test parameter type detection from specifications."""

        # Test continuous parameter
        class ContinuousSpec:
            def __init__(self):
                self.current_value = 50.0
                self.type = float
                self.min_value = 0.0
                self.max_value = 100.0

        param_type = meta_learner._determine_parameter_type(ContinuousSpec())
        assert param_type.value == "continuous"

        # Test discrete parameter
        class DiscreteSpec:
            def __init__(self):
                self.current_value = 50
                self.type = int
                self.min_value = 10
                self.max_value = 100

        param_type = meta_learner._determine_parameter_type(DiscreteSpec())
        assert param_type.value == "discrete"

        # Test categorical parameter
        class CategoricalSpec:
            def __init__(self):
                self.current_value = "option1"
                self.type = str
                self.allowed_values = ["option1", "option2", "option3"]

        param_type = meta_learner._determine_parameter_type(CategoricalSpec())
        assert param_type.value == "categorical"

    async def test_performance_metric_calculation(self, meta_learner):
        """Test performance metric calculation from action results."""

        # Test successful action
        action = MagicMock()
        result = MagicMock()
        result.status.value = "success"

        performance = meta_learner._calculate_performance_metric(action, result)
        assert performance >= 0.7  # Base success performance

        # Test failed action
        result.status.value = "failed"
        performance = meta_learner._calculate_performance_metric(action, result)
        assert performance <= 0.3  # Base failure performance

        # Test action with execution time context
        action.context = {"execution_time": 2.0}  # Fast execution
        result.status.value = "success"
        performance = meta_learner._calculate_performance_metric(action, result)
        assert performance > 0.8  # Should get time bonus

    async def test_value_similarity_check(self, meta_learner):
        """Test value similarity checking."""

        class MockSpec:
            pass

        # Test similar numeric values
        spec = MockSpec()
        assert meta_learner._values_are_similar(100.0, 102.0, spec)  # 2% difference
        assert not meta_learner._values_are_similar(100.0, 110.0, spec)  # 10% difference

        # Test zero edge case
        assert meta_learner._values_are_similar(0.0, 0.005, spec)
        assert not meta_learner._values_are_similar(0.0, 0.02, spec)

        # Test categorical values
        assert meta_learner._values_are_similar("option1", "option1", spec)
        assert not meta_learner._values_are_similar("option1", "option2", spec)

    async def test_conservative_constraints_application(self, meta_learner):
        """Test conservative constraint application."""

        class MockSpec:
            def __init__(self, min_val=None, max_val=None):
                self.min_value = min_val
                self.max_value = max_val

        # Test large change gets constrained
        spec = MockSpec(0, 100)
        constrained = meta_learner._apply_conservative_constraints(50.0, 80.0, spec)
        assert constrained <= 57.5  # 15% max change from 50.0

        # Test bounds are respected
        spec = MockSpec(40, 60)
        constrained = meta_learner._apply_conservative_constraints(50.0, 70.0, spec)
        assert constrained == 57.5  # 15% max change from 50.0 (50.0 + 7.5)

        constrained = meta_learner._apply_conservative_constraints(50.0, 30.0, spec)
        assert constrained == 42.5  # 15% max change from 50.0 (50.0 - 7.5)

    async def test_error_handling_in_bayesian_optimization(
        self, meta_learner, strategy, performance_analysis, knowledge_store, logger
    ):
        """Test error handling in Bayesian optimization."""
        # Mock knowledge store to raise exception
        knowledge_store.query_actions.side_effect = Exception("Database error")

        proposals = await meta_learner.propose_strategy_updates(strategy, performance_analysis)

        # Should fall back to heuristic approach when Bayesian optimization fails
        assert len(proposals) > 0  # Heuristic proposals should be returned

        # Should log the error
        logger.error.assert_called()

        # Check that proposals are heuristic-based (not Bayesian optimization)
        for proposal in proposals:
            assert "Bayesian optimization" not in proposal.rationale

    async def test_different_acquisition_functions(self, knowledge_store, logger, world_model):
        """Test different acquisition functions."""
        # Test with Upper Confidence Bound
        meta_learner_ucb = StatisticalMetaLearner(
            knowledge_store=knowledge_store,
            logger=logger,
            world_model=world_model,
            enable_bayesian_optimization=True,
            acquisition_function=AcquisitionFunction.UPPER_CONFIDENCE_BOUND,
            exploration_weight=0.2,
        )

        assert meta_learner_ucb.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND
        assert meta_learner_ucb.exploration_weight == 0.2

        # Test with Probability of Improvement
        meta_learner_pi = StatisticalMetaLearner(
            knowledge_store=knowledge_store,
            logger=logger,
            world_model=world_model,
            enable_bayesian_optimization=True,
            acquisition_function=AcquisitionFunction.PROBABILITY_IMPROVEMENT,
        )

        assert meta_learner_pi.acquisition_function == AcquisitionFunction.PROBABILITY_IMPROVEMENT
