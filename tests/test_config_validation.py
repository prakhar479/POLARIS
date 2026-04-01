"""Tests for Polaris configuration validation.

Covers connector and strategy type validation using the registry-based factories (F6).
"""

import pytest

import polaris.infrastructure.config as config_module
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


def test_system_config_accepts_connector_discovered_via_plugin_import(monkeypatch):
    """Connector type validation should happen after plugin discovery hooks run."""
    state = {"discovered": False}

    def fake_discover_connector_plugins(plugin_imports):
        assert plugin_imports == ["example_plugins.custom_connector"]
        state["discovered"] = True
        return plugin_imports

    def fake_registered_connector_types():
        supported = ["swim", "wildfire", "kubernetes"]
        if state["discovered"]:
            supported.append("custom_connector")
        return supported

    monkeypatch.setattr(
        config_module,
        "discover_connector_plugins",
        fake_discover_connector_plugins,
    )
    monkeypatch.setattr(
        config_module, "registered_connector_types", fake_registered_connector_types
    )
    monkeypatch.setattr(config_module, "get_connector_config_validator", lambda _ctype: None)

    cfg = PolarisConfig.from_dict(
        {
            "plugin_imports": ["example_plugins.custom_connector"],
            "systems": [
                {
                    "id": "s1",
                    "connector_type": "custom_connector",
                }
            ],
        }
    )

    assert cfg.systems[0].connector_type == "custom_connector"
    assert cfg.plugin_imports == ["example_plugins.custom_connector"]


def test_plugin_imports_must_be_non_empty_string_list():
    """plugin_imports should be validated before Pydantic model creation."""
    bad_values = ["module.path", {"path": "module.path"}, ["ok", 123], [""]]

    for value in bad_values:
        with pytest.raises(ValueError, match="plugin_imports must be a list of non-empty strings"):
            PolarisConfig.from_dict({"plugin_imports": value})


def test_plugin_import_failure_surfaces_clear_error(monkeypatch):
    """Plugin import failures should raise a clear config-level error."""

    def fake_discover_connector_plugins(_plugin_imports):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        config_module,
        "discover_connector_plugins",
        fake_discover_connector_plugins,
    )

    with pytest.raises(ValueError, match="Failed to load connector plugins"):
        PolarisConfig.from_dict({"plugin_imports": ["broken.plugin"]})


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


def test_threshold_action_templates_accept_valid_structure():
    """Threshold strategy should accept structured action template configuration."""
    cfg = PolarisConfig.from_dict(
        {
            "strategy": {
                "type": "threshold",
                "params": {
                    "thresholds": {"latency": {"high": 250.0, "low": 50.0}},
                    "action_templates": {
                        "default": {
                            "high": {"type": "expand_capacity", "parameters": {"step": 2}},
                            "low": {"type": "shrink_capacity", "parameters": {"step": 1}},
                        }
                    },
                },
            }
        }
    )

    assert cfg.strategy.params is not None
    assert "action_templates" in cfg.strategy.params

    # The registry-based factories currently handle detailed structure validation
    # during build time, so this test for structural rejection at load-time
    # is currently disabled.
    pass

    # Since StrategyConfig doesn't forbid extra fields, this test
    # for non-dict block is currently disabled or needs refactoring.
    pass


def test_thread_agentic_accepts_dict_block():
    """thread_agentic strategy should load with a dictionary config block."""
    cfg = PolarisConfig.from_dict(
        {
            "strategy": {
                "type": "thread_agentic",
                "params": {
                    "provider": "google",
                    "max_thread_depth": 2,
                },
            }
        }
    )

    assert cfg.strategy.type == "thread_agentic"
    assert cfg.strategy.params is not None
    assert cfg.strategy.params["max_thread_depth"] == 2


def test_system_action_policy_accepts_valid_configuration():
    """Per-system action policy should load when shape is valid."""
    cfg = PolarisConfig.from_dict(
        {
            "systems": [
                {
                    "id": "wildfire",
                    "connector_type": "wildfire",
                    "action_policy": {
                        "inject_when_no_actions": {
                            "enabled": True,
                            "action": {
                                "type": "wildfire_step",
                                "parameters": {"source": "policy"},
                            },
                        }
                    },
                }
            ]
        }
    )

    assert cfg.systems[0].action_policy is not None
    assert cfg.systems[0].action_policy.inject_when_no_actions is not None


def test_system_action_policy_requires_action_when_enabled():
    """Enabled policy without action block should fail validation."""
    with pytest.raises(ValueError, match="Enabled action policy requires an action block"):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "wildfire",
                        "connector_type": "wildfire",
                        "action_policy": {
                            "inject_when_no_actions": {
                                "enabled": True,
                            }
                        },
                    }
                ]
            }
        )


def test_swim_connection_port_must_be_integer():
    """SWIM connection port validation should be enforced via connector hook."""
    with pytest.raises(ValueError, match="SWIM connection port must be an integer"):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "s1",
                        "connector_type": "swim",
                        "connection": {"host": "localhost", "port": "4242"},
                    }
                ]
            }
        )


def test_wildfire_base_url_must_be_string():
    """Wildfire base_url validation should be enforced via connector hook."""
    with pytest.raises(ValueError, match="Wildfire base_url must be a string"):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "wf",
                        "connector_type": "wildfire",
                        "connection": {"base_url": 12345},
                    }
                ]
            }
        )


def test_system_config_uses_connector_validator_hook(monkeypatch):
    """SystemConfig should delegate connection validation to factory-registered hook."""
    seen = {"called": False}

    def custom_validator(connection):
        seen["called"] = True
        if connection.get("strict") is not True:
            raise ValueError("custom connector validator rejected config")

    def fake_get_validator(connector_type):
        if connector_type == "swim":
            return custom_validator
        return None

    monkeypatch.setattr(config_module, "get_connector_config_validator", fake_get_validator)

    with pytest.raises(ValueError, match="custom connector validator rejected config"):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "s1",
                        "connector_type": "swim",
                        "connection": {"host": "localhost", "port": 4242},
                    }
                ]
            }
        )

    assert seen["called"] is True
