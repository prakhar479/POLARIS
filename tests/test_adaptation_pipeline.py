"""Tests for AdaptationPipeline with multiple actions."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.strategy import AdaptationContext
from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from tests.conftest import MockConnector, MockLogger, MockMetricsCollector, MockStrategy


@pytest.fixture
def pipeline(mock_strategy, mock_logger, mock_metrics):
    config = Mock()
    config.get.return_value = {"enabled": True}
    return AdaptationPipeline(
        strategy=mock_strategy,
        knowledge_store=AsyncMock(),
        world_model=AsyncMock(),
        event_bus=AsyncMock(),
        logger=mock_logger,
        metrics=mock_metrics,
        config=config,
    )


@pytest.mark.asyncio
async def test_pipeline_executes_multiple_actions(pipeline, mock_strategy, mock_connector):
    """Test that the pipeline executes all actions returned by the strategy."""
    # Setup strategy to return two actions
    action1 = AdaptationAction(action_id="1", action_type="scale_up", target_system="test-system")
    action2 = AdaptationAction(action_id="2", action_type="adjust_qos", target_system="test-system")

    mock_strategy.assess = AsyncMock(return_value=[action1, action2])
    mock_strategy.on_action_executed = AsyncMock()

    # Setup connector mocks
    mock_connector.validate_action = AsyncMock(return_value=True)
    mock_connector.execute_action = AsyncMock(
        side_effect=[
            ExecutionResult(action_id="1", status=ExecutionStatus.SUCCESS, result_data={}),
            ExecutionResult(action_id="2", status=ExecutionStatus.SUCCESS, result_data={}),
        ]
    )

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )

    result = await pipeline.run(state, mock_connector)

    assert result is True
    assert mock_connector.execute_action.call_count == 2
    assert mock_strategy.on_action_executed.call_count == 2


@pytest.mark.asyncio
async def test_pipeline_handles_partial_failures(pipeline, mock_strategy, mock_connector):
    """Test that one failing action doesn't stop others."""
    action1 = AdaptationAction(action_id="1", action_type="fail", target_system="test-system")
    action2 = AdaptationAction(action_id="2", action_type="success", target_system="test-system")
    mock_strategy.assess = AsyncMock(return_value=[action1, action2])

    mock_connector.validate_action = AsyncMock(return_value=True)
    mock_connector.execute_action = AsyncMock(
        side_effect=[
            Exception("Execution failed"),
            ExecutionResult(action_id="2", status=ExecutionStatus.SUCCESS, result_data={}),
        ]
    )

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )

    result = await pipeline.run(state, mock_connector)

    assert result is True  # Should still be True because at least one action was attempted
    assert mock_connector.execute_action.call_count == 2
    # Verify both were attempted
    assert any(log[0] == "error" for log in pipeline._logger.logs)


@pytest.mark.asyncio
async def test_pipeline_handles_validation_failure(pipeline, mock_strategy, mock_connector):
    """Test that validation failure skips execution of that specific action."""
    action1 = AdaptationAction(action_id="1", action_type="invalid", target_system="test-system")
    action2 = AdaptationAction(action_id="2", action_type="valid", target_system="test-system")
    mock_strategy.assess = AsyncMock(return_value=[action1, action2])

    mock_connector.validate_action = AsyncMock(side_effect=[False, True])
    mock_connector.execute_action = AsyncMock(
        return_value=ExecutionResult(action_id="2", status=ExecutionStatus.SUCCESS, result_data={})
    )

    state = SystemState(
        system_id="test-system",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )

    result = await pipeline.run(state, mock_connector)

    assert result is True
    mock_connector.execute_action.assert_called_once()
    assert mock_connector.execute_action.call_args[0][0].action_id == "2"
