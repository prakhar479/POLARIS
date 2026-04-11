"""Tests for connector plugin loading helpers."""

from types import ModuleType

import pytest

from polaris.core.factory_plugins import ConnectorPluginLoader


def _build_loader(mock_logger):
    captured = {
        "connector_factory_registrations": 0,
        "connector_validator_registrations": 0,
    }

    def register_connector_factory(_name, _factory):
        captured["connector_factory_registrations"] += 1

    def register_connector_config_validator(_name, _validator):
        captured["connector_validator_registrations"] += 1

    def register_tool_factory(*_args, **_kwargs):
        return None

    loader = ConnectorPluginLoader(
        register_connector_factory=register_connector_factory,
        register_connector_config_validator=register_connector_config_validator,
        register_tool_factory=register_tool_factory,
        entry_point_group="polaris.connectors",
        logger=mock_logger,
    )
    return loader, captured


def test_discover_explicit_plugins_registers_hook_and_deduplicates(monkeypatch, mock_logger):
    loader, captured = _build_loader(mock_logger)
    hook_calls = {"count": 0, "tool_callback_present": False}

    module = ModuleType("example_plugin")

    def register_polaris_plugins(
        register_connector_factory,
        register_connector_config_validator,
        register_tool_factory=None,
    ):
        hook_calls["count"] += 1
        hook_calls["tool_callback_present"] = register_tool_factory is not None
        register_connector_factory("example", lambda *_args, **_kwargs: None)
        register_connector_config_validator("example", lambda _cfg: None)

    module.register_polaris_plugins = register_polaris_plugins

    monkeypatch.setattr(
        "polaris.core.factory_plugins.importlib.import_module",
        lambda _path: module,
    )

    first = loader.discover_explicit_plugins(["example.plugin", "example.plugin"])
    second = loader.discover_explicit_plugins([" example.plugin "])

    assert first == ["example.plugin"]
    assert second == []
    assert hook_calls == {"count": 1, "tool_callback_present": True}
    assert captured == {
        "connector_factory_registrations": 1,
        "connector_validator_registrations": 1,
    }


def test_discover_explicit_plugins_ignores_blank_paths(monkeypatch, mock_logger):
    loader, _captured = _build_loader(mock_logger)
    called = {"value": False}

    def _unexpected_import(_path):
        called["value"] = True
        raise RuntimeError("import should not be called")

    monkeypatch.setattr("polaris.core.factory_plugins.importlib.import_module", _unexpected_import)

    loaded = loader.discover_explicit_plugins(["", "   "])

    assert loaded == []
    assert called["value"] is False


def test_discover_explicit_plugins_rejects_invalid_hook_signature(monkeypatch, mock_logger):
    loader, _captured = _build_loader(mock_logger)
    module = ModuleType("bad_plugin")

    def register_polaris_plugins(not_the_expected_name):
        _ = not_the_expected_name

    module.register_polaris_plugins = register_polaris_plugins

    monkeypatch.setattr(
        "polaris.core.factory_plugins.importlib.import_module", lambda _path: module
    )

    with pytest.raises(TypeError, match="must accept keyword parameters"):
        loader.discover_explicit_plugins(["bad.plugin"])


def test_discover_explicit_plugins_accepts_side_effect_only_module(monkeypatch, mock_logger):
    loader, captured = _build_loader(mock_logger)
    module = ModuleType("side_effect_plugin")

    monkeypatch.setattr(
        "polaris.core.factory_plugins.importlib.import_module",
        lambda _path: module,
    )

    loaded = loader.discover_explicit_plugins(["side.effect.plugin"])

    assert loaded == ["side.effect.plugin"]
    assert captured == {
        "connector_factory_registrations": 0,
        "connector_validator_registrations": 0,
    }


def test_discover_entry_points_loads_callable_plugin_once(monkeypatch, mock_logger):
    loader, captured = _build_loader(mock_logger)
    calls = {"count": 0}

    def plugin_callable(register_connector_factory, register_connector_config_validator):
        calls["count"] += 1
        register_connector_factory("example", lambda *_args, **_kwargs: None)
        register_connector_config_validator("example", lambda _cfg: None)

    class EntryPoint:
        group = "polaris.connectors"
        name = "example"
        module = "example_plugin"

        def load(self):
            return plugin_callable

    monkeypatch.setattr(
        "polaris.core.factory_plugins.importlib.metadata.entry_points",
        lambda group: [EntryPoint(), EntryPoint()],
    )

    loader.discover_entry_points()
    loader.discover_entry_points()

    assert calls["count"] == 1
    assert captured == {
        "connector_factory_registrations": 1,
        "connector_validator_registrations": 1,
    }


def test_discover_entry_points_logs_load_failure_and_continues(monkeypatch, mock_logger):
    loader, captured = _build_loader(mock_logger)

    def plugin_callable(register_connector_factory, register_connector_config_validator):
        register_connector_factory("example", lambda *_args, **_kwargs: None)
        register_connector_config_validator("example", lambda _cfg: None)

    class FailingEntryPoint:
        group = "polaris.connectors"
        name = "broken"
        module = "broken_plugin"

        def load(self):
            raise RuntimeError("boom")

    class WorkingEntryPoint:
        group = "polaris.connectors"
        name = "working"
        module = "working_plugin"

        def load(self):
            return plugin_callable

    monkeypatch.setattr(
        "polaris.core.factory_plugins.importlib.metadata.entry_points",
        lambda group: [FailingEntryPoint(), WorkingEntryPoint()],
    )

    loader.discover_entry_points()

    assert captured == {
        "connector_factory_registrations": 1,
        "connector_validator_registrations": 1,
    }
    warning_messages = [message for level, message, _ctx in mock_logger.logs if level == "warning"]
    assert any(
        "Skipping connector plugin entry point after load failure" in msg
        for msg in warning_messages
    )
