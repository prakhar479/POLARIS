"""Tests for interactive init wizard behavior."""

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
            "openai",
            "gpt-4o",
            "thread_agentic",
        ],
    )

    output_path = tmp_path / "generated.yaml"
    result = init_module.run_init_cli(["--output", str(output_path)])

    assert result == 0
    content = output_path.read_text()
    assert 'connector_type: "custom_connector"' in content
    assert 'type: "thread_agentic"' in content


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
            "openai",
            "gpt-4o",
            "",  # fallback to threshold
        ],
    )

    output_path = tmp_path / "generated_defaults.yaml"
    result = init_module.run_init_cli(["--output", str(output_path)])

    assert result == 0
    content = output_path.read_text()
    assert 'connector_type: "kubernetes"' in content
    assert 'type: "threshold"' in content
