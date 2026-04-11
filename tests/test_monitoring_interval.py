"""Tests for monitoring interval configuration and CLI semantics (F7)."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.core.models import HealthStatus, SystemState
from polaris.core.monitoring_loop import MonitoringLoop
from polaris.core.polaris import Polaris
from polaris.infrastructure.config import PolarisConfig


def test_monitoring_interval_from_config(mock_logger, mock_metrics):
    """monitoring.interval_seconds in config should set the base interval."""
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 15}})

    polaris = Polaris(config=cfg, logger=mock_logger, metrics=mock_metrics)
    # Attribute is normalized to float
    assert polaris._monitoring_interval == 15.0


def test_monitoring_interval_cli_override_positive(mock_logger, mock_metrics):
    """CLI monitoring_interval override should take precedence when positive."""
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 15}})

    polaris = Polaris(
        config=cfg,
        cli_overrides={"monitoring_interval": 5},
        logger=mock_logger,
        metrics=mock_metrics,
    )

    assert polaris._monitoring_interval == 5.0


def test_monitoring_interval_invalid_value_raises_error(mock_logger, mock_metrics):
    """Non-numeric monitoring_interval should raise ValueError."""
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 10}})

    with pytest.raises(ValueError, match="monitoring.interval_seconds must be a number"):
        Polaris(
            config=cfg,
            cli_overrides={"monitoring_interval": "not-a-number"},
            logger=mock_logger,
            metrics=mock_metrics,
        )


def test_monitoring_interval_non_positive_raises_error(mock_logger, mock_metrics):
    """Zero or negative monitoring_interval should raise ValueError."""
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 10}})

    with pytest.raises(ValueError, match="monitoring.interval_seconds must be > 0"):
        Polaris(
            config=cfg,
            cli_overrides={"monitoring_interval": 0},
            logger=mock_logger,
            metrics=mock_metrics,
        )

    with pytest.raises(ValueError, match="monitoring.interval_seconds must be > 0"):
        Polaris(
            config=cfg,
            cli_overrides={"monitoring_interval": -5},
            logger=mock_logger,
            metrics=mock_metrics,
        )


def _build_monitoring_loop(config, mock_logger, mock_metrics, interval_seconds=10.0):
    registry = Mock()
    registry.all.return_value = []
    registry.get_contract.return_value = None

    pipeline = Mock()
    pipeline.run = AsyncMock(return_value=False)

    reloader = Mock()
    reloader.maybe_reload = AsyncMock(return_value=None)

    event_bus = Mock()
    event_bus.publish = AsyncMock(return_value=None)

    return MonitoringLoop(
        registry=registry,
        adaptation_pipeline=pipeline,
        config_reloader=reloader,
        knowledge_store=None,
        world_model=None,
        event_bus=event_bus,
        logger=mock_logger,
        metrics=mock_metrics,
        interval_seconds=interval_seconds,
        config=config,
    )


def test_system_collection_interval_uses_global_floor(mock_logger, mock_metrics):
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {"interval_seconds": 10},
            "systems": [
                {
                    "id": "slow",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 30},
                },
                {
                    "id": "fast",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 3},
                },
            ],
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=10.0)

    assert loop._resolve_system_collection_interval("slow") == 30.0
    assert loop._resolve_system_collection_interval("fast") == 10.0
    assert loop._resolve_system_collection_interval("unknown-system") == 10.0


def test_connector_timeout_resolution_with_global_and_per_system_overrides(
    mock_logger, mock_metrics
):
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {"interval_seconds": 10, "connector_timeout_seconds": 20},
            "systems": [
                {
                    "id": "slow",
                    "connector_type": "unknown",
                    "monitoring": {
                        "collection_interval": 30,
                        "connector_timeout_seconds": 45,
                    },
                },
                {
                    "id": "default",
                    "connector_type": "unknown",
                },
            ],
        }
    )

    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=10.0)

    assert loop._resolve_system_connector_timeout("slow") == 45.0
    assert loop._resolve_system_connector_timeout("default") == 20.0
    assert loop._resolve_system_connector_timeout("missing") == 20.0


def test_system_due_check_respects_effective_interval(mock_logger, mock_metrics):
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {"interval_seconds": 10},
            "systems": [
                {
                    "id": "slow",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 30},
                },
                {
                    "id": "fast",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 2},
                },
            ],
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=10.0)

    now = datetime.now(timezone.utc)
    loop._last_collection_at["slow"] = now
    loop._last_collection_at["fast"] = now

    assert not loop._is_due_for_collection("slow", now + timedelta(seconds=29))
    assert loop._is_due_for_collection("slow", now + timedelta(seconds=30))

    # fast has collection_interval=2, but effective cadence is floored by global interval=10.
    assert not loop._is_due_for_collection("fast", now + timedelta(seconds=9))
    assert loop._is_due_for_collection("fast", now + timedelta(seconds=10))


@pytest.mark.asyncio
async def test_monitoring_loop_skips_not_due_systems(monkeypatch, mock_logger, mock_metrics):
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {"interval_seconds": 1},
            "systems": [
                {
                    "id": "fast",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 1},
                },
                {
                    "id": "slow",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 60},
                },
            ],
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=1.0)

    fast_connector = Mock()
    fast_connector.get_system_id = AsyncMock(return_value="fast")
    slow_connector = Mock()
    slow_connector.get_system_id = AsyncMock(return_value="slow")

    loop._registry.all.return_value = [fast_connector, slow_connector]
    loop._process_system = AsyncMock(
        return_value={"systems_processed": 1, "adaptations_executed": 0}
    )
    loop._last_collection_at["slow"] = datetime.now(timezone.utc)

    async def fake_sleep(_seconds):
        loop._running = False

    monkeypatch.setattr("polaris.core.monitoring_loop.asyncio.sleep", fake_sleep)

    await loop.run()

    assert loop._process_system.await_count == 1
    called_system_id = loop._process_system.await_args_list[0].args[0]
    assert called_system_id == "fast"


@pytest.mark.asyncio
async def test_process_system_telemetry_timeout_records_timeout_metric(mock_logger, mock_metrics):
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {
                "interval_seconds": 1,
                "connector_timeout_seconds": 0.01,
            }
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=1.0)

    async def slow_collect() -> SystemState:
        await asyncio.sleep(0.05)
        return SystemState(
            system_id="timeout-system",
            timestamp=datetime.now(timezone.utc),
            metrics={},
            health_status=HealthStatus.HEALTHY,
        )

    connector = Mock()
    connector.collect_telemetry = slow_collect

    result = await loop._process_system("timeout-system", connector)

    assert result == {"systems_processed": 0, "adaptations_executed": 0}
    assert loop._pipeline.run.await_count == 0
    assert any(
        call[0] == "increment" and call[1] == "polaris.monitoring.timeouts"
        for call in mock_metrics.metrics
    )


@pytest.mark.asyncio
async def test_process_system_pipeline_timeout_keeps_telemetry_processed(mock_logger, mock_metrics):
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {
                "interval_seconds": 1,
                "connector_timeout_seconds": 0.01,
            }
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=1.0)

    async def fast_collect() -> SystemState:
        return SystemState(
            system_id="pipeline-timeout-system",
            timestamp=datetime.now(timezone.utc),
            metrics={},
            health_status=HealthStatus.HEALTHY,
        )

    async def slow_pipeline(*_args, **_kwargs) -> bool:
        await asyncio.sleep(0.05)
        return False

    connector = Mock()
    connector.collect_telemetry = fast_collect
    loop._pipeline.run = AsyncMock(side_effect=slow_pipeline)

    result = await loop._process_system("pipeline-timeout-system", connector)

    assert result == {"systems_processed": 1, "adaptations_executed": 0}
    assert loop._event_bus.publish.await_count == 1
    assert any(
        call[0] == "increment" and call[1] == "polaris.monitoring.timeouts"
        for call in mock_metrics.metrics
    )
