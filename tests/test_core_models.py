"""Tests for core domain models."""

import pytest
from datetime import datetime, timezone
from polaris.core.models import (
    MetricValue, SystemState, AdaptationAction, ExecutionResult,
    HealthStatus, ExecutionStatus
)


class TestMetricValue:
    """Test MetricValue model."""
    
    def test_metric_value_creation(self):
        """Test basic metric value creation."""
        metric = MetricValue("cpu_usage", 75.0, "percent")
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.0
        assert metric.unit == "percent"
        assert isinstance(metric.timestamp, datetime)
        assert metric.tags == {}
    
    def test_metric_value_with_tags(self):
        """Test metric value with tags."""
        tags = {"host": "server1", "region": "us-east"}
        metric = MetricValue("cpu_usage", 75.0, "percent", tags=tags)
        
        assert metric.tags == tags
    
    def test_metric_value_immutable(self):
        """Test that metric values are immutable."""
        metric = MetricValue("cpu_usage", 75.0)
        
        with pytest.raises(AttributeError):
            metric.value = 80.0


class TestSystemState:
    """Test SystemState model."""
    
    def test_system_state_creation(self):
        """Test basic system state creation."""
        timestamp = datetime.now(timezone.utc)
        metrics = {
            "cpu": MetricValue("cpu", 50.0, "percent"),
            "memory": MetricValue("memory", 60.0, "percent")
        }
        
        state = SystemState(
            system_id="test-system",
            timestamp=timestamp,
            metrics=metrics,
            health_status=HealthStatus.HEALTHY
        )
        
        assert state.system_id == "test-system"
        assert state.timestamp == timestamp
        assert state.metrics == metrics
        assert state.health_status == HealthStatus.HEALTHY
        assert state.metadata == {}
    
    def test_system_state_with_metadata(self):
        """Test system state with metadata."""
        metadata = {"version": "1.0", "environment": "prod"}
        state = SystemState(
            system_id="test-system",
            timestamp=datetime.now(timezone.utc),
            metrics={},
            health_status=HealthStatus.HEALTHY,
            metadata=metadata
        )
        
        assert state.metadata == metadata


class TestAdaptationAction:
    """Test AdaptationAction model."""
    
    def test_adaptation_action_creation(self):
        """Test basic adaptation action creation."""
        action = AdaptationAction(
            action_id="test-action",
            action_type="scale_up",
            target_system="test-system",
            parameters={"instances": 2}
        )
        
        assert action.action_id == "test-action"
        assert action.action_type == "scale_up"
        assert action.target_system == "test-system"
        assert action.parameters == {"instances": 2}
        assert action.priority == 0
        assert isinstance(action.created_at, datetime)
    
    def test_adaptation_action_auto_id(self):
        """Test automatic ID generation."""
        action = AdaptationAction(
            action_id="",
            action_type="scale_up",
            target_system="test-system",
            parameters={}
        )
        
        assert action.action_id != ""
        assert len(action.action_id) > 0
    
    def test_adaptation_action_validation(self):
        """Test action validation."""
        # Empty action_type should raise ValueError
        with pytest.raises(ValueError, match="action_type is required"):
            AdaptationAction(
                action_id="test",
                action_type="",
                target_system="test-system",
                parameters={}
            )
        
        # Empty target_system should raise ValueError
        with pytest.raises(ValueError, match="target_system is required"):
            AdaptationAction(
                action_id="test",
                action_type="scale_up",
                target_system="",
                parameters={}
            )
    
    def test_adaptation_action_none_parameters(self):
        """Test that None parameters are converted to empty dict."""
        action = AdaptationAction(
            action_id="test",
            action_type="scale_up",
            target_system="test-system",
            parameters=None
        )
        
        assert action.parameters == {}


class TestExecutionResult:
    """Test ExecutionResult model."""
    
    def test_execution_result_creation(self):
        """Test basic execution result creation."""
        result = ExecutionResult(
            action_id="test-action",
            status=ExecutionStatus.SUCCESS,
            result_data={"message": "Success"}
        )
        
        assert result.action_id == "test-action"
        assert result.status == ExecutionStatus.SUCCESS
        assert result.result_data == {"message": "Success"}
        assert result.error_message is None
        assert isinstance(result.completed_at, datetime)
    
    def test_execution_result_with_error(self):
        """Test execution result with error."""
        result = ExecutionResult(
            action_id="test-action",
            status=ExecutionStatus.FAILED,
            result_data={},
            error_message="Connection failed",
            execution_time_ms=1500
        )
        
        assert result.status == ExecutionStatus.FAILED
        assert result.error_message == "Connection failed"
        assert result.execution_time_ms == 1500


class TestEnums:
    """Test enum values."""
    
    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    def test_execution_status_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.PARTIAL.value == "partial"
        assert ExecutionStatus.TIMEOUT.value == "timeout"