"""State and concurrency hardening tests for factory registries."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import polaris.core.factories as factories


def _run_many(count, fn):
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _idx: fn(), range(count)))


@pytest.fixture(autouse=True)
def _reset_factories_state_between_tests():
    factories._reset_factory_state_for_tests()
    yield
    factories._reset_factory_state_for_tests()


def test_reset_factory_state_clears_registries_and_loader():
    factories.register_connector_factory("test_connector", lambda *_args, **_kwargs: None)
    factories.register_connector_config_validator("test_connector", lambda _cfg: None)
    factories.register_strategy_factory("test_strategy", lambda *_args, **_kwargs: None)
    factories._STATE.plugin_loader = object()
    factories._STATE.factories_registered = True

    factories._reset_factory_state_for_tests()

    assert factories._STATE.connector_factories == {}
    assert factories._STATE.connector_config_validators == {}
    assert factories._STATE.strategy_factories == {}
    assert factories._STATE.plugin_loader is None
    assert factories._STATE.factories_registered is False


def test_lazy_registration_is_thread_safe_and_idempotent(monkeypatch):
    calls = {"connectors": 0, "strategies": 0, "entry_points": 0}

    def fake_register_connectors():
        calls["connectors"] += 1

    def fake_register_strategies():
        calls["strategies"] += 1

    class FakeLoader:
        def discover_entry_points(self):
            calls["entry_points"] += 1

    monkeypatch.setattr(
        factories, "_register_default_connector_factories", fake_register_connectors
    )
    monkeypatch.setattr(factories, "_register_default_strategy_factories", fake_register_strategies)
    monkeypatch.setattr(factories, "_get_plugin_loader", lambda: FakeLoader())

    _run_many(64, factories.registered_connector_types)

    assert calls == {"connectors": 1, "strategies": 1, "entry_points": 1}
    assert factories._STATE.factories_registered is True


def test_reset_allows_reinitialization(monkeypatch):
    calls = {"connectors": 0, "strategies": 0, "entry_points": 0}

    def fake_register_connectors():
        calls["connectors"] += 1

    def fake_register_strategies():
        calls["strategies"] += 1

    class FakeLoader:
        def discover_entry_points(self):
            calls["entry_points"] += 1

    monkeypatch.setattr(
        factories, "_register_default_connector_factories", fake_register_connectors
    )
    monkeypatch.setattr(factories, "_register_default_strategy_factories", fake_register_strategies)
    monkeypatch.setattr(factories, "_get_plugin_loader", lambda: FakeLoader())

    factories.registered_connector_types()
    factories._reset_factory_state_for_tests()
    factories.registered_connector_types()

    assert calls == {"connectors": 2, "strategies": 2, "entry_points": 2}
