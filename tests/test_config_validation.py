"""Tests for Polaris configuration validation.

Covers connector and strategy type validation using the registry-based factories (F6).
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


def test_thread_agentic_requires_dict_block():
    """thread_agentic strategy must receive a dictionary config block."""
    with pytest.raises(ValueError, match="thread_agentic"):
        PolarisConfig.from_dict(
            {
                "strategy": {
                    "type": "thread_agentic",
                    "thread_agentic": "invalid",
                }
            }
        )


def test_thread_agentic_accepts_dict_block():
    """thread_agentic strategy should load with a dictionary config block."""
    cfg = PolarisConfig.from_dict(
        {
            "strategy": {
                "type": "thread_agentic",
                "thread_agentic": {
                    "provider": "google",
                    "max_thread_depth": 2,
                },
            }
        }
    )

    assert cfg.strategy.type == "thread_agentic"
    assert cfg.strategy.thread_agentic is not None
    assert cfg.strategy.thread_agentic["max_thread_depth"] == 2
