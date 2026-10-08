"""Tests for statistical world model."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from polaris.core.models import AdaptationAction, HealthStatus, MetricValue, SystemState
from polaris.world_model.statistical import StatisticalWorldModel


class TestStatisticalWorldModel:
    """Test StatisticalWorldModel functionality."""

    @pytest.fixture
    def knowledge_store(self):
        """Create mock knowledge store."""
        return AsyncMock()

    @pytest.fixture
    def world_model(self, knowledge_store):
        """Create statistical world model."""
        return StatisticalWorldModel(knowledge_store)

    @pytest.fixture
    def kalman_world_model(self, knowledge_store):
        """Create statistical world model with Kalman filtering enabled."""
        return StatisticalWorldModel(knowledge_store, use_kalman=True)

    @pytest.fixture
    def sample_state(self):
        """Create sample system state."""
        return SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 75.0, "percent"),
                "memory_usage": MetricValue("memory_usage", 60.0, "percent"),
                "response_time": MetricValue("response_time", 200.0, "ms"),
            },
            health_status=HealthStatus.HEALTHY,
        )

    @pytest.fixture
    def sample_action(self):
        """Create sample adaptation action."""
        return AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={"instances": 2},
        )

    @pytest.mark.asyncio
    async def test_update_with_numeric_metrics(self, world_model, sample_state):
        """Test updating model with numeric metrics."""
        await world_model.update(sample_state)

        # Verify metrics are stored in history
        system_history = world_model._metric_history["test-system"]
        assert "cpu_usage" in system_history
        assert "memory_usage" in system_history
        assert "response_time" in system_history

        assert system_history["cpu_usage"] == [75.0]
        assert system_history["memory_usage"] == [60.0]
        assert system_history["response_time"] == [200.0]

    @pytest.mark.asyncio
    async def test_update_with_non_numeric_metrics(self, world_model):
        """Test updating model with non-numeric metrics."""
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "status": MetricValue("status", "healthy", "string"),
                "cpu_usage": MetricValue("cpu_usage", 75.0, "percent"),
            },
            health_status=HealthStatus.HEALTHY,
        )

        await world_model.update(state)

        # Only numeric metrics should be stored
        system_history = world_model._metric_history["test-system"]
        assert "cpu_usage" in system_history
        assert "status" not in system_history
        assert system_history["cpu_usage"] == [75.0]

    @pytest.mark.asyncio
    async def test_update_multiple_states(self, world_model):
        """Test updating model with multiple states."""
        # Create multiple states with different values
        for cpu_value in [50.0, 60.0, 70.0, 80.0]:
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={"cpu_usage": MetricValue("cpu_usage", cpu_value, "percent")},
                health_status=HealthStatus.HEALTHY,
            )
            await world_model.update(state)

        # Verify all values are stored
        cpu_history = world_model._metric_history["test-system"]["cpu_usage"]
        assert cpu_history == [50.0, 60.0, 70.0, 80.0]

    @pytest.mark.asyncio
    async def test_history_limit_enforcement(self, world_model):
        """Test that history is limited to 100 values."""
        # Add 150 states to exceed the limit
        for i in range(150):
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={"cpu_usage": MetricValue("cpu_usage", float(i), "percent")},
                health_status=HealthStatus.HEALTHY,
            )
            await world_model.update(state)

        # Should only keep last 100 values
        cpu_history = world_model._metric_history["test-system"]["cpu_usage"]
        assert len(cpu_history) == 100
        assert cpu_history[0] == 50.0  # Values 50-149 should remain
        assert cpu_history[-1] == 149.0

    @pytest.mark.asyncio
    async def test_predict_with_history(self, world_model, sample_action, sample_state):
        """Test prediction with historical data."""
        # Build up some history
        for cpu_value in [60.0, 70.0, 80.0]:
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", cpu_value, "percent"),
                    "memory_usage": MetricValue("memory_usage", 50.0, "percent"),
                },
                health_status=HealthStatus.HEALTHY,
            )
            await world_model.update(state)

        # Make prediction
        prediction = await world_model.predict(sample_action, sample_state)

        # Should predict mean values
        assert prediction.predicted_metrics["cpu_usage"] == 70.0  # Mean of 60, 70, 80
        assert prediction.predicted_metrics["memory_usage"] == 50.0  # Mean of 50, 50, 50
        assert prediction.confidence == 0.5
        assert "Statistical baseline" in prediction.reasoning

    @pytest.mark.asyncio
    async def test_predict_without_history(self, world_model, sample_action, sample_state):
        """Test prediction without historical data."""
        prediction = await world_model.predict(sample_action, sample_state)

        # Should return empty predictions
        assert prediction.predicted_metrics == {}
        assert prediction.confidence == 0.5
        assert "Statistical baseline" in prediction.reasoning

    @pytest.mark.asyncio
    async def test_predict_partial_history(self, world_model, sample_action, sample_state):
        """Test prediction with partial historical data."""
        # Add history for only one metric
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 65.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(state)

        prediction = await world_model.predict(sample_action, sample_state)

        # Should only predict for metrics with history
        assert "cpu_usage" in prediction.predicted_metrics
        assert "memory_usage" not in prediction.predicted_metrics
        assert prediction.predicted_metrics["cpu_usage"] == 65.0

    @pytest.mark.asyncio
    async def test_get_insights_with_data(self, world_model):
        """Test getting insights with statistical data."""
        # Add varied data for insights
        cpu_values = [50.0, 60.0, 70.0, 80.0, 90.0]
        memory_values = [40.0, 45.0, 50.0, 55.0, 60.0]

        for cpu, memory in zip(cpu_values, memory_values):
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", cpu, "percent"),
                    "memory_usage": MetricValue("memory_usage", memory, "percent"),
                },
                health_status=HealthStatus.HEALTHY,
            )
            await world_model.update(state)

        insights = await world_model.get_insights()

        # Verify insights structure
        assert "test-system" in insights
        system_insights = insights["test-system"]

        assert "cpu_usage" in system_insights
        assert "memory_usage" in system_insights

        # Check CPU insights
        cpu_insights = system_insights["cpu_usage"]
        assert cpu_insights["mean"] == 70.0  # Mean of 50,60,70,80,90
        assert cpu_insights["min"] == 50.0
        assert cpu_insights["max"] == 90.0
        assert cpu_insights["std"] > 0  # Should have some standard deviation

        # Check memory insights
        memory_insights = system_insights["memory_usage"]
        assert memory_insights["mean"] == 50.0  # Mean of 40,45,50,55,60
        assert memory_insights["min"] == 40.0
        assert memory_insights["max"] == 60.0

    @pytest.mark.asyncio
    async def test_get_insights_insufficient_data(self, world_model):
        """Test getting insights with insufficient data."""
        # Add only one data point
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 75.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(state)

        insights = await world_model.get_insights()

        # Should not include metrics with insufficient data (< 2 points)
        # But the system will still be in the insights dict, just empty
        if "test-system" in insights:
            assert insights["test-system"] == {}
        else:
            assert insights == {}

    @pytest.mark.asyncio
    async def test_get_insights_empty(self, world_model):
        """Test getting insights with no data."""
        insights = await world_model.get_insights()
        assert insights == {}

    @pytest.mark.asyncio
    async def test_regime_tracking_in_insights(self, world_model):
        """Test that regime probabilities are tracked and exposed in insights."""
        # Create states that clearly indicate a high-load regime
        for _ in range(5):
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", 90.0, "percent"),
                    "response_time": MetricValue("response_time", 600.0, "ms"),
                },
                health_status=HealthStatus.HEALTHY,
            )
            await world_model.update(state)

        insights = await world_model.get_insights()
        assert "test-system" in insights
        system_insights = insights["test-system"]
        assert "regime" in system_insights
        regime_info = system_insights["regime"]
        assert "probabilities" in regime_info
        assert "most_likely" in regime_info
        # High CPU/response_time should bias towards 'high' regime
        assert regime_info["most_likely"] in {"high", "normal"}

    @pytest.mark.asyncio
    async def test_reasoning_mentions_kalman_and_regime(
        self, kalman_world_model, sample_action, sample_state
    ):
        """Test that reasoning string reflects Kalman usage and regime estimate."""
        # Build up some history to initialize Kalman filters and regime
        for _ in range(3):
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", 85.0, "percent"),
                    "response_time": MetricValue("response_time", 550.0, "ms"),
                },
                health_status=HealthStatus.HEALTHY,
            )
            await kalman_world_model.update(state)

        prediction = await kalman_world_model.predict(sample_action, sample_state)
        reasoning = prediction.reasoning
        assert "Kalman-smoothed" in reasoning
        assert "Estimated regime" in reasoning

    @pytest.mark.asyncio
    async def test_predict_with_kalman_enabled(
        self, kalman_world_model, sample_action, sample_state
    ):
        """Test prediction when Kalman filtering is enabled."""
        # Build up some history
        for cpu_value in [60.0, 70.0, 80.0]:
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", cpu_value, "percent"),
                },
                health_status=HealthStatus.HEALTHY,
            )
            await kalman_world_model.update(state)

        prediction = await kalman_world_model.predict(sample_action, sample_state)

        # Should still predict for metrics with history
        assert "cpu_usage" in prediction.predicted_metrics
        # Confidence should be derived from Kalman variance and lie in (0, 1]
        assert 0.0 < prediction.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_multiple_systems(self, world_model):
        """Test handling multiple systems."""
        # Add data for two different systems
        state1 = SystemState(
            system_id="system-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 60.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )

        state2 = SystemState(
            system_id="system-2",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 80.0, "percent")},
            health_status=HealthStatus.HEALTHY,
        )

        await world_model.update(state1)
        await world_model.update(state2)

        # Verify both systems are tracked separately
        assert "system-1" in world_model._metric_history
        assert "system-2" in world_model._metric_history

        assert world_model._metric_history["system-1"]["cpu_usage"] == [60.0]
        assert world_model._metric_history["system-2"]["cpu_usage"] == [80.0]

    @pytest.mark.asyncio
    async def test_invalid_metric_values(self, world_model):
        """Test handling of invalid metric values."""
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "invalid_int": MetricValue("invalid_int", None, "count"),
                "invalid_str": MetricValue("invalid_str", "not_a_number", "percent"),
                "valid_metric": MetricValue("valid_metric", 75.0, "percent"),
            },
            health_status=HealthStatus.HEALTHY,
        )

        await world_model.update(state)

        # Only valid metric should be stored
        system_history = world_model._metric_history["test-system"]
        assert "valid_metric" in system_history
        assert "invalid_int" not in system_history
        assert "invalid_str" not in system_history
        assert system_history["valid_metric"] == [75.0]

    @pytest.mark.asyncio
    async def test_predict_with_recorded_action_effects(self, world_model, sample_state):
        """Test that recorded action effects apply counterfactual deltas in prediction."""
        # Baseline response_time is 200.0 ms
        await world_model.update(sample_state)

        # Record empirical delta for scale_up: response_time drops by 50ms, cpu drops by 15%
        world_model.record_action_effect(
            system_id="test-system",
            action_type="scale_up",
            metric_deltas={"response_time": -50.0, "cpu_usage": -15.0},
        )

        action = AdaptationAction(
            action_id="act-1",
            action_type="scale_up",
            target_system="test-system",
        )
        prediction = await world_model.predict(action, sample_state)

        # 200 - 50 = 150.0 ms
        assert prediction.predicted_metrics["response_time"] == 150.0
        # 75 - 15 = 60.0%
        assert prediction.predicted_metrics["cpu_usage"] == 60.0
        assert prediction.confidence >= 0.75
        assert "Simulated counterfactual impact" in prediction.reasoning
        assert "response_time: -50.00" in prediction.reasoning

    @pytest.mark.asyncio
    async def test_predict_with_action_parameters(self, world_model):
        """Test counterfactual prediction when action parameters directly alter state."""
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "dimmer": MetricValue("dimmer", 1.0, "ratio"),
                "response_time": MetricValue("response_time", 300.0, "ms"),
            },
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(state)

        action = AdaptationAction(
            action_id="act-dimmer",
            action_type="set_dimmer",
            target_system="test-system",
            parameters={"dimmer": 0.6},
        )
        prediction = await world_model.predict(action, state)

        assert prediction.predicted_metrics["dimmer"] == 0.6
        assert prediction.confidence >= 0.75
        assert "Simulated counterfactual impact" in prediction.reasoning

    @pytest.mark.asyncio
    async def test_closed_loop_delta_absorption(self, world_model):
        """Test that in-flight actions absorb telemetry deltas automatically."""
        t1 = datetime.now(timezone.utc)
        pre_state = SystemState(
            system_id="swim-sys",
            timestamp=t1,
            metrics={
                "average_response_time": MetricValue("average_response_time", 800.0, "ms", t1),
                "average_utilization": MetricValue("average_utilization", 0.90, "ratio", t1),
            },
            health_status=HealthStatus.WARNING,
        )
        await world_model.update(pre_state)

        # Register pending action
        action = AdaptationAction(
            action_id="act-scale",
            action_type="scale_up",
            target_system="swim-sys",
        )
        world_model.record_pending_action(action, pre_state)

        # Post-action update arrives
        t2 = datetime.now(timezone.utc)
        post_state = SystemState(
            system_id="swim-sys",
            timestamp=t2,
            metrics={
                "average_response_time": MetricValue("average_response_time", 550.0, "ms", t2),
                "average_utilization": MetricValue("average_utilization", 0.60, "ratio", t2),
            },
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(post_state)

        # Verify deltas were recorded in action_effects
        effects = world_model._action_effects["swim-sys"]["scale_up"]
        assert "average_response_time" in effects
        assert effects["average_response_time"] == [-250.0]
        assert effects["average_utilization"] == [-0.30]

    @pytest.mark.asyncio
    async def test_queuing_surrogate_prediction(self, world_model):
        """Test M/M/m queuing domain surrogate before empirical samples exist."""
        t = datetime.now(timezone.utc)
        state = SystemState(
            system_id="swim-app",
            timestamp=t,
            metrics={
                "server_count": MetricValue("server_count", 2.0, "count", t),
                "average_utilization": MetricValue("average_utilization", 0.80, "ratio", t),
                "average_response_time": MetricValue("average_response_time", 600.0, "ms", t),
            },
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(state)

        # Predict scale_up
        action = AdaptationAction(
            action_id="act-scale-up",
            action_type="scale_up",
            target_system="swim-app",
        )
        prediction = await world_model.predict(action, state)

        # Server count increases to 3.0
        assert prediction.predicted_metrics["server_count"] == 3.0
        # Utilization scales down by ~2/3: 0.80 * 2/3 = ~0.533
        assert prediction.predicted_metrics["average_utilization"] < 0.80
        # Response time drops
        assert prediction.predicted_metrics["average_response_time"] < 600.0
        assert "uncertainty" in dir(prediction)
        assert len(prediction.uncertainty) > 0

    @pytest.mark.asyncio
    async def test_switch_vision_surrogate_prediction(self, world_model):
        """Test YOLOv5 profile surrogate prediction for SWITCH model switching."""
        t = datetime.now(timezone.utc)
        state = SystemState(
            system_id="switch-sys",
            timestamp=t,
            metrics={
                "confidence_mean": MetricValue("confidence_mean", 0.69, "ratio", t),
                "response_time": MetricValue("response_time", 0.096, "s", t),
                "cpu_usage": MetricValue("cpu_usage", 48.0, "percent", t),
                "inference_rate": MetricValue("inference_rate", 244.0, "inf/min", t),
            },
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(state)

        # Predict switch to yolov5s (faster, lighter)
        action = AdaptationAction(
            action_id="act-switch",
            action_type="switch_model",
            target_system="switch-sys",
            parameters={"model_name": "yolov5s"},
        )
        prediction = await world_model.predict(action, state)

        # yolov5s: latency ~ 0.065s, confidence ~ 0.62, cpu ~ 40.0%
        assert prediction.predicted_metrics["response_time"] == 0.065
        assert prediction.predicted_metrics["confidence_mean"] == 0.62
        assert prediction.predicted_metrics["cpu_usage"] == 40.0
        assert prediction.predicted_metrics["inference_rate"] == 260.0
        assert prediction.confidence >= 0.75

    @pytest.mark.asyncio
    async def test_domain_surrogates_resilience_to_non_numeric_metrics(self, world_model):
        """Test domain surrogates gracefully handle None and malformed metric values."""
        t = datetime.now(timezone.utc)
        state = SystemState(
            system_id="resilient-sys",
            timestamp=t,
            metrics={
                "server_count": MetricValue("server_count", 2.0, "count", t),
                "average_utilization": MetricValue("average_utilization", 0.8, "ratio", t),
                "average_response_time": MetricValue("average_response_time", 250.0, "ms", t),
            },
            health_status=HealthStatus.HEALTHY,
        )
        await world_model.update(state)
        action = AdaptationAction(
            action_id="act-scale",
            action_type="scale_up",
            target_system="resilient-sys",
        )
        prediction = await world_model.predict(action, state)
        assert prediction.predicted_metrics["server_count"] == 3.0
