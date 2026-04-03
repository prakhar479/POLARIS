"""Tests for interactive init wizard behavior."""

import yaml

import polaris.cli.init as init_module


def _set_input_answers(monkeypatch, answers):
    values = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(values))


def test_init_cli_uses_registered_connector_and_strategy_types(monkeypatch, tmp_path):
    monkeypatch.setattr(
        init_module,
        "registered_connector_types",
        lambda: ["swim", "custom_connector"],
    )
    monkeypatch.setattr(
        init_module,
        "registered_strategy_types",
        lambda: ["hybrid", "thread_agentic"],
    )
    _set_input_answers(
        monkeypatch,
        [
            "custom_connector",
            "7",
            "y",
            "thread_agentic",
            "openai",
            "gpt-4o",
        ],
    )

    output_path = tmp_path / "generated.yaml"
    result = init_module.run_init_cli(["--output", str(output_path)])

    assert result == 0
    content = output_path.read_text()
    parsed = yaml.safe_load(content)
    assert 'connector_type: "custom_connector"' in content
    assert 'type: "thread_agentic"' in content
    assert 'formats: ["json", "csv"]' in content
    assert 'output_dir: "metrics"' in content
    assert parsed["strategy"]["type"] == "thread_agentic"
    assert parsed["strategy"]["params"]["provider"] == "openai"


def test_init_cli_falls_back_to_registered_defaults_for_unknown_values(monkeypatch, tmp_path):
    monkeypatch.setattr(
        init_module,
        "registered_connector_types",
        lambda: ["kubernetes"],
    )
    monkeypatch.setattr(
        init_module,
        "registered_strategy_types",
        lambda: ["threshold"],
    )
    _set_input_answers(
        monkeypatch,
        [
            "",  # fallback to kubernetes
            "5",
            "n",
            "",  # fallback to threshold
        ],
    )

    output_path = tmp_path / "generated_defaults.yaml"
    result = init_module.run_init_cli(["--output", str(output_path)])

    assert result == 0
    content = output_path.read_text()
    parsed = yaml.safe_load(content)
    assert 'connector_type: "kubernetes"' in content
    assert 'type: "threshold"' in content
    assert 'namespace: "default"' in content
    assert "in_cluster: false" in content
    assert parsed["systems"][0]["connection"]["namespace"] == "default"
    assert parsed["strategy"]["type"] == "threshold"
