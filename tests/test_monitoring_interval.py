"""Tests for monitoring interval configuration and CLI semantics (F7)."""

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


def test_monitoring_interval_invalid_value_falls_back(mock_logger, mock_metrics):
    """Non-numeric monitoring_interval should fall back to 30 seconds."""
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 10}})

    polaris = Polaris(
        config=cfg,
        cli_overrides={"monitoring_interval": "not-a-number"},
        logger=mock_logger,
        metrics=mock_metrics,
    )

    assert polaris._monitoring_interval == 30.0


def test_monitoring_interval_non_positive_falls_back(mock_logger, mock_metrics):
    """Zero or negative monitoring_interval should fall back to 30 seconds."""
    cfg = PolarisConfig.from_dict({"monitoring": {"interval_seconds": 10}})

    polaris_zero = Polaris(
        config=cfg,
        cli_overrides={"monitoring_interval": 0},
        logger=mock_logger,
        metrics=mock_metrics,
    )
    assert polaris_zero._monitoring_interval == 30.0

    polaris_negative = Polaris(
        config=cfg,
        cli_overrides={"monitoring_interval": -5},
        logger=mock_logger,
        metrics=mock_metrics,
    )
    assert polaris_negative._monitoring_interval == 30.0
