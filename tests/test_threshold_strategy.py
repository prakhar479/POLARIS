"""Tests for threshold reactive strategy."""

from datetime import datetime, timedelta, timezone

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.strategies.threshold import ThresholdReactiveStrategy


class TestThresholdReactiveStrategy:
    """Test ThresholdReactiveStrategy functionality."""

    @pytest.fixture
    def strategy(self, mock_logger, mock_metrics):
        """Create strategy with mock dependencies."""
        return ThresholdReactiveStrategy(
            thresholds={
                "cpu_usage": {"high": 80.0, "low": 20.0},
                "memory_usage": {"high": 85.0, "low": 25.0},
            },
            cooldown_seconds=60,
            logger=mock_logger,
            metrics=mock_metrics,
        )

    @pytest.fixture
    def context(self):
        """Create adaptation context."""
        return AdaptationContext(system_id="test-system", historical_states=[])

    def test_strategy_initialization(self, mock_logger, mock_metrics):
        """Test strategy initialization."""
        strategy = ThresholdReactiveStrategy(
            thresholds={"cpu": {"high": 90.0}},
            cooldown_seconds=30,
            logger=mock_logger,
            metrics=mock_metrics,
        )

        assert strategy.thresholds == {"cpu": {"high": 90.0}}
        assert strategy.cooldown_seconds == 30
        assert strategy.logger == mock_logger
        assert strategy.metrics == mock_metrics

    def test_default_thresholds(self):
        """Test default threshold configuration."""
        strategy = ThresholdReactiveStrategy()

        expected_thresholds = {
            "cpu_usage": {"high": 80.0, "low": 20.0},
            "memory_usage": {"high": 85.0, "low": 25.0},
        }
        assert strategy.thresholds == expected_thresholds
        assert strategy.cooldown_seconds == 60

    @pytest.mark.asyncio
    async def test_high_threshold_exceeded(self, strategy, context):
        """Test action creation when high threshold is exceeded."""
        # Create state with high CPU usage
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 85.0, "percent"),  # Above 80% threshold
                "memory_usage": MetricValue("memory_usage", 50.0, "percent"),
            },
            health_status=HealthStatus.HEALTHY,
        )

        action = await strategy.assess(state, context)

        assert action is not None
        assert action.action_type == "scale_up"
        assert action.target_system == "test-system"
        assert action.parameters["metric"] == "cpu_usage"
        assert action.parameters["current_value"] == 85.0
        assert action.parameters["threshold"] == 80.0

    @pytest.mark.asyncio
    async def test_low_threshold_breached(self, strategy, context):
        """Test action creation when low threshold is breached."""
        # Create state with low CPU usage
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 15.0, "percent"),  # Below 20% threshold
                "memory_usage": MetricValue("memory_usage", 50.0, "percent"),
            },
            health_status=HealthStatus.HEALTHY,
        )

        action = await strategy.assess(state, context)

        assert action is not None
        assert action.action_type == "scale_down"
        assert action.target_system == "test-system"
        assert action.parameters["metric"] == "cpu_usage"
        assert action.parameters["current_value"] == 15.0
        assert action.parameters["threshold"] == 20.0

    @pytest.mark.asyncio
    async def test_no_threshold_crossed(self, strategy, context):
        """Test no action when thresholds are not crossed."""
        # Create state with normal values
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 50.0, "percent"),  # Between thresholds
                "memory_usage": MetricValue("memory_usage", 60.0, "percent"),  # Between thresholds
            },
            health_status=HealthStatus.HEALTHY,
        )

        action = await strategy.assess(state, context)

        assert action is None

    @pytest.mark.asyncio
    async def test_cooldown_period(self, strategy, context):
        """Test cooldown period prevents rapid adaptations."""
        # Create state that exceeds threshold
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 85.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )

        # First assessment should return action
        action1 = await strategy.assess(state, context)
        assert action1 is not None

        # Second assessment immediately after should return None (cooldown)
        action2 = await strategy.assess(state, context)
        assert action2 is None

    @pytest.mark.asyncio
    async def test_cooldown_expires(self, strategy, context):
        """Test that cooldown expires after configured time."""
        # Mock the last adaptation time to be in the past
        past_time = datetime.now(timezone.utc) - timedelta(seconds=120)  # 2 minutes ago
        strategy._last_adaptation["test-system"] = past_time

        # Create state that exceeds threshold
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 85.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )

        # Should return action since cooldown has expired
        action = await strategy.assess(state, context)
        assert action is not None

    @pytest.mark.asyncio
    async def test_unknown_metric(self, strategy, context):
        """Test handling of metrics not in threshold configuration."""
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"unknown_metric": MetricValue("unknown_metric", 100.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )

        action = await strategy.assess(state, context)
        assert action is None

    @pytest.mark.asyncio
    async def test_invalid_metric_value(self, strategy, context):
        """Test handling of non-numeric metric values."""
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", "invalid", "percent")},
            health_status=HealthStatus.HEALTHY,
        )

        action = await strategy.assess(state, context)
        assert action is None

    @pytest.mark.asyncio
    async def test_server_count_logic(self, strategy, context):
        """Test inverted logic for server_count metric."""
        # Add server_count threshold
        strategy.thresholds["server_count"] = {"high": 10, "low": 2}

        # Low server count should trigger scale_up
        state_low = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "server_count": MetricValue("server_count", 1, "count")
            },  # Below low threshold
            health_status=HealthStatus.HEALTHY,
        )

        action_low = await strategy.assess(state_low, context)
        assert action_low is not None
        assert action_low.action_type == "scale_up"

        # Reset cooldown for next test
        strategy._last_adaptation.clear()

        # High server count should trigger scale_down
        state_high = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "server_count": MetricValue("server_count", 15, "count")
            },  # Above high threshold
            health_status=HealthStatus.HEALTHY,
        )

        action_high = await strategy.assess(state_high, context)
        assert action_high is not None
        assert action_high.action_type == "scale_down"

    @pytest.mark.asyncio
    async def test_on_action_executed(self, strategy):
        """Test action execution tracking."""
        action = AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={},
        )

        result = ExecutionResult(
            action_id="test-action", status=ExecutionStatus.SUCCESS, result_data={}
        )

        initial_count = strategy._adaptation_count
        initial_success = strategy._success_count

        await strategy.on_action_executed(action, result)

        assert strategy._adaptation_count == initial_count + 1
        assert strategy._success_count == initial_success + 1

    def test_get_tunable_parameters(self, strategy):
        """Test getting tunable parameters."""
        params = strategy.get_tunable_parameters()

        # Should include threshold parameters
        assert "thresholds.cpu_usage.high" in params
        assert "thresholds.cpu_usage.low" in params
        assert "thresholds.memory_usage.high" in params
        assert "thresholds.memory_usage.low" in params
        assert "cooldown_seconds" in params

        # Check parameter specs
        cpu_high_spec = params["thresholds.cpu_usage.high"]
        assert cpu_high_spec.current_value == 80.0
        assert cpu_high_spec.type == float

    @pytest.mark.asyncio
    async def test_update_parameter(self, strategy):
        """Test parameter updates."""
        # Update cooldown
        success = await strategy.update_parameter("cooldown_seconds", 120)
        assert success
        assert strategy.cooldown_seconds == 120

        # Update threshold
        success = await strategy.update_parameter("thresholds.cpu_usage.high", 90.0)
        assert success
        assert strategy.thresholds["cpu_usage"]["high"] == 90.0

        # Invalid parameter
        success = await strategy.update_parameter("invalid_param", 100)
        assert not success

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, strategy):
        """Test performance metrics calculation."""
        # Initially no adaptations
        metrics = await strategy.get_performance_metrics()
        assert metrics["success_rate"] == 0.0

        # Simulate some adaptations
        strategy._adaptation_count = 10
        strategy._success_count = 8

        metrics = await strategy.get_performance_metrics()
        assert metrics["success_rate"] == 0.8
        assert metrics["total_adaptations"] == 10.0
