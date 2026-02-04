"""Tests for Polaris configuration validation.

Covers connector and strategy type validation using the registry-based
factories (F6).
"""

import pytest

from polaris.infrastructure.config import PolarisConfig


def test_system_config_valid_connector_types():
    """Systems with supported connector types should load without error."""
    cfg = PolarisConfig.from_dict(
        {
            "systems": [
                {"id": "s1", "connector_type": "swim"},
                {"id": "s2", "connector_type": "wildfire"},
            ]
        }
    )

    assert len(cfg.systems) == 2
    assert cfg.systems[0].connector_type == "swim"
    assert cfg.systems[1].connector_type == "wildfire"


def test_system_config_invalid_connector_type():
    """Unknown connector types should raise a clear validation error."""
    with pytest.raises(ValueError, match="Unsupported connector type"):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {"id": "s1", "connector_type": "not_real"},
                ]
            }
        )


def test_strategy_config_invalid_type():
    """Unknown strategy types should raise a clear validation error."""
    with pytest.raises(ValueError, match="Unsupported strategy type"):
        PolarisConfig.from_dict(
            {
                "strategy": {
                    "type": "unknown_strategy_type",
                }
            }
        )
