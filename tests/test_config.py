"""Extra configuration tests."""

import os

import pytest

from polaris.infrastructure.config import PolarisConfig, StrategyConfig, load_config


def test_populate_params_non_dict():
    res = StrategyConfig.populate_params("not a dict")
    assert res == "not a dict"


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
