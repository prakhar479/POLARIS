"""Tests for statistical world model."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from polaris.world_model.statistical import StatisticalWorldModel
from polaris.core.models import SystemState, AdaptationAction, MetricValue, HealthStatus


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
    def sample_state(self):
        """Create sample system state."""
        return SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 75.0, "percent"),
                "memory_usage": MetricValue("memory_usage", 60.0, "percent"),
                "response_time": MetricValue("response_time", 200.0, "ms")
            },
            health_status=HealthStatus.HEALTHY
        )
    
    @pytest.fixture
    def sample_action(self):
        """Create sample adaptation action."""
        return AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={"instances": 2}
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
                "cpu_usage": MetricValue("cpu_usage", 75.0, "percent")
            },
            health_status=HealthStatus.HEALTHY
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
        for i, cpu_value in enumerate([50.0, 60.0, 70.0, 80.0]):
            state = SystemState(
                system_id="test-system",
                timestamp=datetime.now(timezone.utc),
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", cpu_value, "percent")
                },
                health_status=HealthStatus.HEALTHY
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
                metrics={
                    "cpu_usage": MetricValue("cpu_usage", float(i), "percent")
                },
                health_status=HealthStatus.HEALTHY
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
                    "memory_usage": MetricValue("memory_usage", 50.0, "percent")
                },
                health_status=HealthStatus.HEALTHY
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
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 65.0, "percent")
            },
            health_status=HealthStatus.HEALTHY
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
                    "memory_usage": MetricValue("memory_usage", memory, "percent")
                },
                health_status=HealthStatus.HEALTHY
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
            metrics={
                "cpu_usage": MetricValue("cpu_usage", 75.0, "percent")
            },
            health_status=HealthStatus.HEALTHY
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
    async def test_multiple_systems(self, world_model):
        """Test handling multiple systems."""
        # Add data for two different systems
        state1 = SystemState(
            system_id="system-1",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 60.0, "percent")},
            health_status=HealthStatus.HEALTHY
        )
        
        state2 = SystemState(
            system_id="system-2",
            timestamp=datetime.now(timezone.utc),
            metrics={"cpu_usage": MetricValue("cpu_usage", 80.0, "percent")},
            health_status=HealthStatus.HEALTHY
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
                "valid_metric": MetricValue("valid_metric", 75.0, "percent")
            },
            health_status=HealthStatus.HEALTHY
        )
        
        await world_model.update(state)
        
        # Only valid metric should be stored
        system_history = world_model._metric_history["test-system"]
        assert "valid_metric" in system_history
        assert "invalid_int" not in system_history
        assert "invalid_str" not in system_history
        assert system_history["valid_metric"] == [75.0]