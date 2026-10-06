"""Tests for push-based telemetry ingestion in MonitoringLoop and Polaris coordinator."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.core.events import InMemoryEventBus, TelemetryEvent
from polaris.core.models import (
    AdaptationAction,
    ExecutionResult,
    ExecutionStatus,
    HealthStatus,
    MetricValue,
    SystemState,
)
from polaris.core.monitoring_loop import MonitoringLoop
from polaris.core.polaris import Polaris
from polaris.core.registry import ConnectorRegistry
from polaris.infrastructure.config import PolarisConfig
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.knowledge.memory import InMemoryKnowledgeStore
from tests.conftest import MockLogger


@pytest.fixture
def mock_logger():
    return MockLogger()


@pytest.fixture
def mock_event_bus():
    return InMemoryEventBus()


@pytest.fixture
def knowledge_store():
    return InMemoryKnowledgeStore()


@pytest.mark.asyncio
async def test_monitoring_loop_ingest_telemetry_queues_and_wakes(
    mock_logger, mock_event_bus, knowledge_store
):
    """Pushed telemetry wakes the loop immediately and runs through storage and adaptation."""
    await mock_event_bus.start()
    received_events = []
    mock_event_bus.subscribe(TelemetryEvent, lambda ev: received_events.append(ev))

    registry = ConnectorRegistry()
    connector = Mock()
    connector.get_system_id = AsyncMock(return_value="push-sys")
    connector.validate_action = AsyncMock(return_value=True)
    connector.execute_action = AsyncMock(
        return_value=ExecutionResult(
            action_id="act-1",
            status=ExecutionStatus.SUCCESS,
            result_data={"handled": True},
        )
    )
    from polaris.abstractions.system_contract import SystemContract

    contract = SystemContract(system_id="push-sys", supported_action_types=("act-1",))
    await registry.register(connector, contract=contract)

    pipeline = Mock()
    pipeline.run = AsyncMock(return_value=True)

    world_model = Mock()
    world_model.update = AsyncMock()

    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 60}})
    reloader = Mock()
    reloader.maybe_reload = AsyncMock(return_value=None)

    loop = MonitoringLoop(
        registry=registry,
        adaptation_pipeline=pipeline,
        config_reloader=reloader,
        knowledge_store=knowledge_store,
        world_model=world_model,
        event_bus=mock_event_bus,
        logger=mock_logger,
        metrics=NullMetricsCollector(),
        interval_seconds=60.0,
        config=cfg,
    )

    state = SystemState(
        system_id="push-sys",
        timestamp=datetime.now(timezone.utc),
        metrics={"latency": MetricValue("latency", 42.0)},
        health_status=HealthStatus.HEALTHY,
    )

    # Ingest telemetry
    queued = await loop.ingest_telemetry(state)
    assert queued is True

    # Process pushed state directly
    res = await loop._process_pushed_state(state)
    assert res["systems_processed"] == 1
    assert res["adaptations_executed"] == 1

    # Verify pipeline was invoked immediately with the pushed state
    assert pipeline.run.call_count == 1
    pipeline_state_arg = pipeline.run.call_args[0][0]
    assert pipeline_state_arg.system_id == "push-sys"
    assert pipeline_state_arg.metrics["latency"].value == 42.0

    # Verify knowledge store and world model were updated
    states = await knowledge_store.query_states(
        "push-sys",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    assert len(states) == 1
    assert world_model.update.call_count == 1

    # Verify TelemetryEvent was published
    assert len(received_events) == 1
    assert received_events[0].system_id == "push-sys"


@pytest.mark.asyncio
async def test_monitoring_loop_ingest_telemetry_drops_when_queue_full(
    mock_logger, mock_event_bus, knowledge_store
):
    """When the bounded telemetry queue is saturated, ingest returns False and logs warning."""
    registry = ConnectorRegistry()
    pipeline = Mock()
    reloader = Mock()
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 60}})

    loop = MonitoringLoop(
        registry=registry,
        adaptation_pipeline=pipeline,
        config_reloader=reloader,
        knowledge_store=knowledge_store,
        world_model=None,
        event_bus=mock_event_bus,
        logger=mock_logger,
        metrics=NullMetricsCollector(),
        interval_seconds=60.0,
        config=cfg,
    )

    # Create a tiny 1-element queue to test overflow
    loop._telemetry_queue = asyncio.Queue(maxsize=1)

    state1 = SystemState(
        system_id="overflow-sys",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )
    state2 = SystemState(
        system_id="overflow-sys",
        timestamp=datetime.now(timezone.utc),
        metrics={},
        health_status=HealthStatus.HEALTHY,
    )

    assert await loop.ingest_telemetry(state1) is True
    # Second should drop because queue is full
    assert await loop.ingest_telemetry(state2) is False


@pytest.mark.asyncio
async def test_polaris_ingest_telemetry_delegates_to_monitoring_loop(
    mock_logger, mock_event_bus, knowledge_store
):
    """Polaris.ingest_telemetry routes to active monitoring loop or falls back to knowledge store."""
    polaris = Polaris(
        config=PolarisConfig(),
        knowledge_store=knowledge_store,
        logger=mock_logger,
        event_bus=mock_event_bus,
    )

    state = SystemState(
        system_id="sys-offline",
        timestamp=datetime.now(timezone.utc),
        metrics={"cpu": MetricValue("cpu", 50.0)},
        health_status=HealthStatus.HEALTHY,
    )

    # Before run(): falls back to persisting in knowledge store
    res_before = await polaris.ingest_telemetry(state)
    assert res_before is True
    states = await knowledge_store.query_states(
        "sys-offline",
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2030, 1, 1, tzinfo=timezone.utc),
    )
    assert len(states) == 1

    # Simulate active monitoring loop
    mock_loop = Mock()
    mock_loop.ingest_telemetry = AsyncMock(return_value=True)
    polaris._monitoring_loop = mock_loop

    res_active = await polaris.ingest_telemetry(state)
    assert res_active is True
    assert mock_loop.ingest_telemetry.call_count == 1
    assert polaris.monitoring_loop is mock_loop
