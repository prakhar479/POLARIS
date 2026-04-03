"""Tests for CLI doctor diagnostics and interactive CLI routing."""

import importlib
from pathlib import Path

from polaris.cli import doctor as doctor_cli

cli_main = importlib.import_module("polaris.cli.main")


def test_doctor_basic_config_has_no_failures(tmp_path: Path) -> None:
    """A minimal config should pass doctor checks without hard failures."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    diagnostics = doctor_cli.run_doctor(str(config_file))
    failures = [item for item in diagnostics if item.status == "FAIL"]

    assert failures == []


def test_doctor_reports_missing_placeholder_env(tmp_path: Path, monkeypatch) -> None:
    """Config placeholders must report missing environment variables."""
    monkeypatch.delenv("POLARIS_MISSING_TOKEN", raising=False)

    config_file = tmp_path / "config.yaml"
    config_file.write_text("token: ${POLARIS_MISSING_TOKEN}\n", encoding="utf-8")

    diagnostics = doctor_cli.run_doctor(str(config_file))

    assert any(
        item.status == "FAIL" and "POLARIS_MISSING_TOKEN" in item.message for item in diagnostics
    )


def test_doctor_reports_missing_llm_credentials(tmp_path: Path, monkeypatch) -> None:
    """LLM-enabled configs must report absent provider credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "strategy:",
                "  type: llm_reasoning",
                "  params:",
                "    provider: openai",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostics = doctor_cli.run_doctor(str(config_file))

    assert any(
        item.status == "FAIL" and "strategy.params: missing credentials" in item.message
        for item in diagnostics
    )


def test_doctor_reports_missing_thread_agentic_credentials(tmp_path: Path, monkeypatch) -> None:
    """thread_agentic configs should also report missing provider credentials."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "strategy:",
                "  type: thread_agentic",
                "  params:",
                "    provider: openai",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostics = doctor_cli.run_doctor(str(config_file))

    assert any(
        item.status == "FAIL" and "strategy.params: missing credentials" in item.message
        for item in diagnostics
    )


def test_doctor_flags_legacy_strategy_schema_paths(tmp_path: Path) -> None:
    """Doctor should report deprecated type-keyed strategy blocks with explicit paths."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "strategy:",
                "  type: llm_reasoning",
                "  llm_reasoning:",
                "    provider: openai",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    diagnostics = doctor_cli.run_doctor(str(config_file))

    assert any(
        item.status == "FAIL"
        and "Legacy strategy schema detected" in item.message
        and "strategy.llm_reasoning" in item.message
        for item in diagnostics
    )


def test_main_dispatches_doctor_subcommand(monkeypatch) -> None:
    """`polaris doctor` should dispatch to doctor CLI parser."""
    called = {}

    def fake_run_doctor_cli(argv):
        called["argv"] = list(argv)
        return 17

    monkeypatch.setattr(doctor_cli, "run_doctor_cli", fake_run_doctor_cli)
    monkeypatch.setattr(cli_main.sys, "argv", ["polaris", "doctor", "--config", "cfg.yaml"])

    assert cli_main.main() == 17
    assert called["argv"] == ["--config", "cfg.yaml"]


def test_main_interactive_uses_single_process_runner(tmp_path: Path, monkeypatch) -> None:
    """`--interactive` should run the in-process interactive runner."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    captured = {"coroutine_name": None}

    class FakePolaris:
        def __init__(self, config_path=None, cli_overrides=None):
            self.config_path = config_path
            self.cli_overrides = cli_overrides or {}

    async def fake_run_with_interactive_cli(_polaris):
        return None

    async def fake_run_framework(_polaris):
        return None

    def fake_asyncio_run(coro):
        captured["coroutine_name"] = coro.cr_code.co_name
        coro.close()

    monkeypatch.setattr(cli_main, "Polaris", FakePolaris)
    monkeypatch.setattr(cli_main, "run_with_interactive_cli", fake_run_with_interactive_cli)
    monkeypatch.setattr(cli_main, "run_framework", fake_run_framework)
    monkeypatch.setattr(cli_main.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(
        cli_main.sys,
        "argv",
        ["polaris", "--interactive", "--config", str(config_file)],
    )

    exit_code = cli_main.main()

    assert exit_code == 0
    assert captured["coroutine_name"] == "fake_run_with_interactive_cli"


def test_main_both_uses_split_dashboard_mode(tmp_path: Path, monkeypatch) -> None:
    """`--both` should run the combined dashboard + interactive coroutine."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    captured = {"coroutine_name": None}

    class FakePolaris:
        def __init__(self, config_path=None, cli_overrides=None):
            self.config_path = config_path
            self.cli_overrides = cli_overrides or {}

    async def fake_run_with_dashboard_and_interactive(_polaris, clear_screen=False):
        _ = clear_screen
        return None

    async def fake_run_framework(_polaris):
        return None

    def fake_asyncio_run(coro):
        captured["coroutine_name"] = coro.cr_code.co_name
        coro.close()

    monkeypatch.setattr(cli_main, "Polaris", FakePolaris)
    monkeypatch.setattr(
        cli_main,
        "run_with_dashboard_and_interactive",
        fake_run_with_dashboard_and_interactive,
    )
    monkeypatch.setattr(cli_main, "run_framework", fake_run_framework)
    monkeypatch.setattr(cli_main.asyncio, "run", fake_asyncio_run)
    monkeypatch.setattr(
        cli_main.sys,
        "argv",
        ["polaris", "--both", "--config", str(config_file)],
    )

    exit_code = cli_main.main()

    assert exit_code == 0
    assert captured["coroutine_name"] == "fake_run_with_dashboard_and_interactive"
