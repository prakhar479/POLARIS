"""Extra configuration tests."""

import os

import pytest

from polaris.infrastructure.config import PolarisConfig, StrategyConfig, load_config


def test_strategy_config_rejects_legacy_type_keyed_blocks():
    with pytest.raises(ValueError, match="Extra inputs are not permitted|extra_forbidden"):
        PolarisConfig.from_dict(
            {
                "strategy": {
                    "type": "llm_reasoning",
                    "llm_reasoning": {"provider": "openai"},
                }
            }
        )


def test_strategy_config_hybrid_validation():
    with pytest.raises(ValueError, match="hybrid selection_mode must be one of"):
        StrategyConfig(type="hybrid", params={"selection_mode": "invalid"})

    with pytest.raises(ValueError, match="hybrid min_confidence must be a float"):
        StrategyConfig(type="hybrid", params={"min_confidence": "high"})

    with pytest.raises(ValueError, match="hybrid min_confidence must be a float"):
        StrategyConfig(type="hybrid", params={"min_confidence": 2.0})

    # valid
    cfg = StrategyConfig(type="hybrid", params={"selection_mode": "first", "min_confidence": 0.5})
    assert cfg.params["selection_mode"] == "first"


def test_strategy_config_hybrid_rejects_legacy_sub_strategy_blocks():
    with pytest.raises(ValueError, match="unsupported keys"):
        StrategyConfig(
            type="hybrid",
            params={
                "strategies": [
                    {
                        "type": "threshold",
                        "threshold": {"thresholds": {"cpu": {"high": 80.0}}},
                    }
                ]
            },
        )


def test_strategy_config_threshold_bounds_validation():
    with pytest.raises(ValueError, match="must be greater than low bound"):
        StrategyConfig(
            type="threshold",
            params={"thresholds": {"cpu": {"high": 40.0, "low": 50.0}}},
        )


def test_strategy_config_llm_provider_validation():
    with pytest.raises(ValueError, match="provider must be one of"):
        StrategyConfig(type="agentic_llm", params={"provider": "unknown"})


def test_strategy_config_thread_phi_mode_validation():
    with pytest.raises(ValueError, match="phi_mode"):
        StrategyConfig(type="thread_agentic", params={"phi_mode": "invalid_mode"})


def test_strategy_config_agentic_native_policy_validation():
    with pytest.raises(ValueError, match="native_tools_unsupported_policy"):
        StrategyConfig(
            type="agentic_llm",
            params={"native_tools_unsupported_policy": "unsupported"},
        )


def test_strategy_config_agentic_tool_payload_bound_validation():
    with pytest.raises(ValueError, match="max_tool_result_chars"):
        StrategyConfig(type="agentic_llm", params={"max_tool_result_chars": 0})


def test_strategy_config_multi_agent_tool_payload_bound_validation():
    with pytest.raises(ValueError, match="max_tool_result_chars"):
        StrategyConfig(type="multi_agent", params={"max_tool_result_chars": 0})


def test_strategy_config_rejects_unknown_tool_names_semantically():
    with pytest.raises(ValueError, match="unknown tool name"):
        StrategyConfig(
            type="agentic_llm",
            params={"tools": {"enabled": ["definitely_not_a_tool"]}},
        )


def test_strategy_config_native_tools_known_tool_must_be_enabled():
    with pytest.raises(ValueError, match="not enabled under strategy.params.tools"):
        StrategyConfig(
            type="agentic_llm",
            params={
                "tools": {"enabled": ["get_action_history"]},
                "native_tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_recent_states"},
                    }
                ],
            },
        )


def test_strategy_config_tools_shape_validation():
    with pytest.raises(ValueError, match="enabled must be a list"):
        StrategyConfig(type="multi_agent", params={"tools": {"enabled": "tool-a"}})


def test_normalize_max_concurrent_connectors_invalid():
    res = PolarisConfig.normalize_max_concurrent_connectors("not a dict")
    assert res == "not a dict"

    with pytest.raises(ValueError, match="max_concurrent_connectors must be an integer > 0"):
        PolarisConfig.normalize_max_concurrent_connectors({"max_concurrent_connectors": "invalid"})


def test_from_file_not_found():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        PolarisConfig.from_file("nonexistent.yaml")


def test_from_dict_non_dict():
    with pytest.raises(ValueError, match="Config root must be a dictionary"):
        PolarisConfig.from_dict(["not a dict"])


def test_substitute_env_vars():
    os.environ["TEST_POLARIS_VAR"] = "success"
    try:
        content = "var: ${TEST_POLARIS_VAR}"
        res = PolarisConfig._substitute_env_vars(content)
        assert res == "var: success"

        with pytest.raises(ValueError, match="Environment variable 'MISSING_VAR' not found"):
            PolarisConfig._substitute_env_vars("var: ${MISSING_VAR}")
    finally:
        del os.environ["TEST_POLARIS_VAR"]


def test_load_config(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("max_concurrent_connectors: 5\n")
    cfg = load_config(str(f))
    assert cfg.max_concurrent_connectors == 5


def test_system_collection_interval_validation():
    with pytest.raises(
        ValueError,
        match=r"systems\[\]\.monitoring\.collection_interval",
    ):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "sys-1",
                        "connector_type": "unknown",
                        "monitoring": {"collection_interval": 0},
                    }
                ]
            }
        )

    with pytest.raises(
        ValueError,
        match=r"systems\[\]\.monitoring\.collection_interval",
    ):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "sys-1",
                        "connector_type": "unknown",
                        "monitoring": {"collection_interval": "5"},
                    }
                ]
            }
        )

    cfg = PolarisConfig.from_dict(
        {
            "systems": [
                {
                    "id": "sys-1",
                    "connector_type": "unknown",
                    "monitoring": {"collection_interval": 5},
                }
            ]
        }
    )
    assert cfg.systems[0].monitoring["collection_interval"] == 5


def test_connector_timeout_validation():
    with pytest.raises(
        ValueError,
        match=r"monitoring\.connector_timeout_seconds",
    ):
        PolarisConfig.from_dict(
            {
                "monitoring": {
                    "connector_timeout_seconds": 0,
                }
            }
        )

    with pytest.raises(
        ValueError,
        match=r"systems\[\]\.monitoring\.connector_timeout_seconds",
    ):
        PolarisConfig.from_dict(
            {
                "systems": [
                    {
                        "id": "sys-1",
                        "connector_type": "unknown",
                        "monitoring": {"connector_timeout_seconds": -1},
                    }
                ]
            }
        )

    cfg = PolarisConfig.from_dict(
        {
            "monitoring": {
                "connector_timeout_seconds": 20,
            },
            "systems": [
                {
                    "id": "sys-1",
                    "connector_type": "unknown",
                    "monitoring": {"connector_timeout_seconds": 5},
                }
            ],
        }
    )

    assert cfg.monitoring["connector_timeout_seconds"] == 20
    assert cfg.systems[0].monitoring["connector_timeout_seconds"] == 5
