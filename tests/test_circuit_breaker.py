"""Tests for AdaptationPipeline circuit breaker and fallback strategy."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.core.adaptation_pipeline import AdaptationPipeline
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    SystemState,
)
from tests.conftest import MockConnector, MockLogger, MockMetricsCollector, MockStrategy


@pytest.fixture
def base_state():
    return SystemState(
        system_id="test-sys",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )


@pytest.mark.asyncio
async def test_circuit_breaker_delegates_to_fallback_on_primary_failure(
    base_state, mock_logger, mock_metrics
):
    """When primary strategy fails, fallback strategy is called and actions executed."""
    primary = MockStrategy()
    primary.assess = AsyncMock(side_effect=RuntimeError("Primary LLM connection timeout"))

    fallback = MockStrategy()
    fallback_action = AdaptationAction(
        action_id="fb-1", action_type="scale_up", target_system="test-sys"
    )
    fallback.assess = AsyncMock(return_value=[fallback_action])

    connector = MockConnector("test-sys")
    connector.validate_action = AsyncMock(return_value=True)
    connector.execute_action = AsyncMock(
        return_value=ExecutionResult(
            action_id="fb-1", status=ExecutionStatus.SUCCESS, result_data={}
        )
    )

    config = Mock()
    config.get.return_value = {"enabled": True}

    pipeline = AdaptationPipeline(
        strategy=primary,
        knowledge_store=AsyncMock(),
        world_model=AsyncMock(),
        event_bus=AsyncMock(),
        logger=mock_logger,
        metrics=mock_metrics,
        config=config,
        fallback_strategy=fallback,
        circuit_breaker_threshold=2,
        circuit_breaker_recovery_seconds=10.0,
    )

    # First failure on primary -> delegates to fallback
    result = await pipeline.run(base_state, connector)
    assert result is True
    assert primary.assess.call_count == 1
    assert fallback.assess.call_count == 1
    assert pipeline.consecutive_failures == 1
    assert pipeline.circuit_breaker_state == "CLOSED"


@pytest.mark.asyncio
async def test_circuit_breaker_trips_to_open_after_threshold(base_state, mock_logger, mock_metrics):
    """Circuit breaker opens after threshold consecutive failures."""
    primary = MockStrategy()
    primary.assess = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

    fallback = MockStrategy()
    fallback.assess = AsyncMock(
        return_value=[
            AdaptationAction(action_id="fb-2", action_type="scale_up", target_system="test-sys")
        ]
    )

    connector = MockConnector("test-sys")
    connector.validate_action = AsyncMock(return_value=True)
    connector.execute_action = AsyncMock(
        return_value=ExecutionResult(
            action_id="fb-2", status=ExecutionStatus.SUCCESS, result_data={}
        )
    )

    config = Mock()
    config.get.return_value = {"enabled": True}

    pipeline = AdaptationPipeline(
        strategy=primary,
        knowledge_store=AsyncMock(),
        world_model=AsyncMock(),
        event_bus=AsyncMock(),
        logger=mock_logger,
        metrics=mock_metrics,
        config=config,
        fallback_strategy=fallback,
        circuit_breaker_threshold=2,
        circuit_breaker_recovery_seconds=30.0,
    )

    # Failure 1
    await pipeline.run(base_state, connector)
    assert pipeline.consecutive_failures == 1
    assert pipeline.circuit_breaker_state == "CLOSED"

    # Failure 2: reaches threshold -> trips to OPEN
    await pipeline.run(base_state, connector)
    assert pipeline.consecutive_failures == 2
    assert pipeline.circuit_breaker_state == "OPEN"

    # Call 3: circuit breaker is OPEN, primary is NOT called, fallback runs directly
    primary.assess.reset_mock()
    fallback.assess.reset_mock()

    await pipeline.run(base_state, connector)
    assert primary.assess.call_count == 0  # skipped primary
    assert fallback.assess.call_count == 1  # delegated directly to fallback


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery(base_state, mock_logger, mock_metrics):
    """After recovery seconds pass, breaker enters HALF_OPEN and resets on success."""
    primary = MockStrategy()
    primary.assess = AsyncMock(side_effect=RuntimeError("Temporary error"))

    fallback = MockStrategy()
    fallback.assess = AsyncMock(return_value=[])

    connector = MockConnector("test-sys")
    config = Mock()
    config.get.return_value = {"enabled": True}

    pipeline = AdaptationPipeline(
        strategy=primary,
        knowledge_store=AsyncMock(),
        world_model=AsyncMock(),
        event_bus=AsyncMock(),
        logger=mock_logger,
        metrics=mock_metrics,
        config=config,
        fallback_strategy=fallback,
        circuit_breaker_threshold=1,
        circuit_breaker_recovery_seconds=0.01,  # short recovery window
    )

    # Fail once to open
    await pipeline.run(base_state, connector)
    assert pipeline.circuit_breaker_state == "OPEN"

    # Wait for recovery window
    await asyncio.sleep(0.02)

    # Now make primary succeed
    recovery_action = AdaptationAction(
        action_id="rec-1", action_type="scale_up", target_system="test-sys"
    )
    primary.assess = AsyncMock(return_value=[recovery_action])
    connector.validate_action = AsyncMock(return_value=True)
    connector.execute_action = AsyncMock(
        return_value=ExecutionResult(
            action_id="rec-1", status=ExecutionStatus.SUCCESS, result_data={}
        )
    )

    await pipeline.run(base_state, connector)
    assert pipeline.circuit_breaker_state == "CLOSED"
    assert pipeline.consecutive_failures == 0
