"""Tests for PolarisConfig normalization and component wiring."""

import pytest

from polaris.core.polaris import Polaris
from polaris.infrastructure.config import PolarisConfig
from polaris.knowledge.memory import InMemoryKnowledgeStore
from polaris.world_model.statistical import StatisticalWorldModel


def test_knowledge_store_new_key_memory(mock_logger, mock_metrics):
    """knowledge_store.memory should map to in-memory knowledge store."""
    cfg = PolarisConfig.from_dict(
        {
            "knowledge_store": {
                "type": "memory",
                "memory": {"max_states_per_system": 321},
            }
        }
    )

    polaris = Polaris(config=cfg, logger=mock_logger, metrics=mock_metrics)
    assert isinstance(polaris.knowledge_store, InMemoryKnowledgeStore)
    assert polaris.knowledge_store.max_states == 321


def test_knowledge_legacy_key_is_rejected():
    """Legacy `knowledge` key must fail after migration."""
    with pytest.raises(ValueError, match="knowledge_store"):
        PolarisConfig.from_dict({"knowledge": {"type": "memory"}})


def test_world_model_config_is_applied(mock_logger, mock_metrics):
    """world_model.statistical settings should be passed into model construction."""
    cfg = PolarisConfig.from_dict(
        {
            "world_model": {
                "type": "statistical",
                "statistical": {
                    "use_kalman": True,
                    "window_size": 42,
                },
            }
        }
    )

    polaris = Polaris(config=cfg, logger=mock_logger, metrics=mock_metrics)
    assert isinstance(polaris.world_model, StatisticalWorldModel)
    assert polaris.world_model._use_kalman is True
    assert polaris.world_model._window_size == 42


def test_max_concurrent_connectors_is_normalized():
    """max_concurrent_connectors should parse int values and fail on invalid inputs."""
    cfg_valid = PolarisConfig.from_dict({"max_concurrent_connectors": "3"})
    assert cfg_valid.max_concurrent_connectors == 3

    with pytest.raises(Exception):
        PolarisConfig.from_dict({"max_concurrent_connectors": -1})


def test_unknown_top_level_keys_are_rejected():
    """Unknown top-level keys should fail fast instead of being preserved implicitly."""
    with pytest.raises(Exception) as exc_info:
        PolarisConfig.from_dict({"wildfire": {"always_step_each_cycle": True}})

    assert "wildfire" in str(exc_info.value)
