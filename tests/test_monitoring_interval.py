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


def test_stress_adaptive_cadence_on_health_warning_or_critical(mock_logger, mock_metrics):
    """When adaptive_cadence is enabled, WARNING/CRITICAL health accelerates polling."""
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {
                "interval_seconds": 10,
                "adaptive_cadence": True,
                "stress_multiplier": 0.5,
                "min_adaptive_interval": 2.0,
            },
            "systems": [
                {
                    "id": "stressed-sys",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 10},
                }
            ],
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=10.0)

    # Initial healthy state -> normal interval (10.0s)
    loop._latest_system_health["stressed-sys"] = HealthStatus.HEALTHY
    assert loop._resolve_system_collection_interval("stressed-sys") == 10.0

    # Stressed WARNING state -> accelerated interval (10.0 * 0.5 = 5.0s)
    loop._latest_system_health["stressed-sys"] = HealthStatus.WARNING
    assert loop._resolve_system_collection_interval("stressed-sys") == 5.0

    # Stressed CRITICAL state -> accelerated interval (5.0s)
    loop._latest_system_health["stressed-sys"] = HealthStatus.CRITICAL
    assert loop._resolve_system_collection_interval("stressed-sys") == 5.0

    # Back to HEALTHY -> relaxes back to 10.0s
    loop._latest_system_health["stressed-sys"] = HealthStatus.HEALTHY
    assert loop._resolve_system_collection_interval("stressed-sys") == 10.0


def test_stress_adaptive_cadence_on_high_regime(mock_logger, mock_metrics):
    """When world model high regime probability > 0.5, adaptive cadence accelerates polling."""
    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {
                "interval_seconds": 20,
                "adaptive_cadence": True,
                "stress_multiplier": 0.25,
                "min_adaptive_interval": 3.0,
            },
            "systems": [
                {
                    "id": "regime-sys",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 20},
                }
            ],
        }
    )
    loop = _build_monitoring_loop(cfg, mock_logger, mock_metrics, interval_seconds=20.0)
    mock_world_model = Mock()
    mock_world_model._regime_probs = {"regime-sys": {"low": 0.1, "normal": 0.2, "high": 0.7}}
    loop._world_model = mock_world_model

    # High regime triggers acceleration (20.0 * 0.25 = 5.0s)
    assert loop._resolve_system_collection_interval("regime-sys") == 5.0

    # Minimum adaptive floor test (e.g. min_adaptive_interval = 8.0)
    cfg_floor = PolarisConfig.from_dict(
        {
            "monitoring": {
                "interval_seconds": 10,
                "adaptive_cadence": True,
                "stress_multiplier": 0.2,  # 10 * 0.2 = 2.0, but min is 6.0
                "min_adaptive_interval": 6.0,
            },
            "systems": [
                {
                    "id": "regime-sys",
                    "connector_type": "unknown",
                }
            ],
        }
    )
    loop_floor = _build_monitoring_loop(cfg_floor, mock_logger, mock_metrics, interval_seconds=10.0)
    loop_floor._world_model = mock_world_model
    assert loop_floor._resolve_system_collection_interval("regime-sys") == 6.0
