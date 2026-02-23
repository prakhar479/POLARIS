"""Focused tests for component factory logic and config branching."""

from types import SimpleNamespace

import pytest

from polaris.core.component_builder import ComponentBuilder
from polaris.infrastructure.observability.metrics import SimpleMetricsCollector
from polaris.infrastructure.observability.null_metrics import NullMetricsCollector
from polaris.knowledge import InMemoryKnowledgeStore
from polaris.knowledge.sqlite_store import SQLiteKnowledgeStore
from polaris.strategies import ThresholdReactiveStrategy
from polaris.world_model import StatisticalWorldModel


def _cfg(
    *,
    observability=None,
    knowledge_store=None,
    world_model=None,
    monitoring=None,
):
    return SimpleNamespace(
        observability=observability or {},
        knowledge_store=knowledge_store or {},
        world_model=world_model or {},
        monitoring=monitoring or {},
    )


def test_build_logger_applies_config_and_cli_overrides(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_create_logger(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "polaris.infrastructure.observability.logger.create_logger",
        fake_create_logger,
    )

    config = _cfg(
        observability={
            "logging": {
                "type": "human",
                "level": "WARNING",
                "console": False,
                "use_colors": False,
                "file": True,
                "file_path": "/tmp/from-config.log",
            }
        }
    )
    cli_overrides = {
        "log_format": "structured",
        "log_level": "DEBUG",
        "console_logging": True,
        "log_file": "/tmp/from-cli.log",
    }

    logger = ComponentBuilder.build_logger(config, cli_overrides)

    assert logger is sentinel
    assert captured == {
        "logger_type": "structured",
        "name": "polaris",
        "level": "DEBUG",
        "log_file": "/tmp/from-cli.log",
        "console": True,
        "use_colors": False,
    }


def test_build_metrics_honors_disable_flags():
    config_enabled = _cfg(observability={"metrics": {"enabled": True, "collector_type": "simple"}})
    config_disabled = _cfg(observability={"metrics": {"enabled": False}})
    config_unknown = _cfg(observability={"metrics": {"enabled": True, "collector_type": "other"}})

    assert isinstance(
        ComponentBuilder.build_metrics(config_enabled, {"metrics_enabled": False}),
        NullMetricsCollector,
    )
    assert isinstance(ComponentBuilder.build_metrics(config_disabled, {}), NullMetricsCollector)
    assert isinstance(ComponentBuilder.build_metrics(config_enabled, {}), SimpleMetricsCollector)
    assert isinstance(ComponentBuilder.build_metrics(config_unknown, {}), SimpleMetricsCollector)


def test_build_event_bus_respects_per_component_metrics_toggle(mock_logger, mock_metrics):
    cfg_disabled = _cfg(observability={"metrics": {"components": {"event_bus": False}}})
    cfg_enabled = _cfg(observability={"metrics": {"components": {"event_bus": True}}})

    bus_disabled = ComponentBuilder.build_event_bus(cfg_disabled, mock_metrics, mock_logger)
    bus_enabled = ComponentBuilder.build_event_bus(cfg_enabled, mock_metrics, mock_logger)

    assert bus_disabled._metrics is None
    assert bus_enabled._metrics is mock_metrics


def test_build_knowledge_store_sqlite_uses_nested_config(tmp_path, mock_logger, mock_metrics):
    db_path = str(tmp_path / "polaris.db")
    config = _cfg(
        observability={"metrics": {"components": {"knowledge_store": False}}},
        knowledge_store={
            "type": "sqlite",
            "sqlite": {
                "db_path": db_path,
                "max_states_per_system": 42,
            },
        },
    )

    store = ComponentBuilder.build_knowledge_store(config, mock_logger, mock_metrics)

    assert isinstance(store, SQLiteKnowledgeStore)
    assert store._db_path == db_path
    assert store._max_states == 42
    assert store._metrics is None


def test_build_knowledge_store_unknown_type_falls_back_to_memory(mock_logger):
    config = _cfg(
        knowledge_store={
            "type": "unknown",
            "memory": {"max_states_per_system": 7},
        }
    )

    store = ComponentBuilder.build_knowledge_store(config, mock_logger, None)

    assert isinstance(store, InMemoryKnowledgeStore)
    assert store.max_states == 7
    assert any(
        level == "warning" and "Unknown knowledge store type" in message
        for level, message, _ in mock_logger.logs
    )


def test_build_world_model_applies_defaults_and_sanitizes_window(mock_logger):
    knowledge_store = InMemoryKnowledgeStore()
    config = _cfg(
        observability={"metrics": {"components": {"world_model": False}}},
        world_model={
            "type": "custom",
            "statistical": {
                "use_kalman": True,
                "window_size": "not-an-int",
            },
        },
    )

    world_model = ComponentBuilder.build_world_model(config, knowledge_store, mock_logger, object())

    assert isinstance(world_model, StatisticalWorldModel)
    assert world_model._use_kalman is True
    assert world_model._window_size == 100
    assert world_model._metrics is None
    assert any(
        level == "warning" and "Unknown world model type" in message
        for level, message, _ in mock_logger.logs
    )


def test_build_strategy_handles_missing_and_failing_factories(mock_logger, monkeypatch):
    knowledge_store = InMemoryKnowledgeStore()
    world_model = StatisticalWorldModel(knowledge_store)
    config = _cfg()
    strategy_config = SimpleNamespace(type="custom")

    fallback = ComponentBuilder.build_strategy(
        strategy_config,
        mock_logger,
        None,
        knowledge_store,
        world_model,
        SimpleNamespace(),
        config,
    )
    assert isinstance(fallback, ThresholdReactiveStrategy)

    def bad_factory(*_args, **_kwargs):
        raise RuntimeError("factory failed")

    monkeypatch.setattr("polaris.core.factories.get_strategy_factory", lambda _type: bad_factory)
    fallback_after_error = ComponentBuilder.build_strategy(
        strategy_config,
        mock_logger,
        None,
        knowledge_store,
        world_model,
        SimpleNamespace(),
        config,
    )

    assert isinstance(fallback_after_error, ThresholdReactiveStrategy)
    assert any(
        level == "warning" and "Failed to initialize strategy" in message
        for level, message, _ in mock_logger.logs
    )


def test_build_meta_learner_statistical_defaults_invalid_acquisition(mock_logger, mock_metrics):
    knowledge_store = InMemoryKnowledgeStore()
    world_model = StatisticalWorldModel(knowledge_store)
    config = _cfg()
    meta_cfg = {
        "type": "statistical",
        "statistical": {
            "acquisition_function": "not-real",
            "exploration_weight": 0.25,
            "min_samples_for_optimization": 7,
            "conservative_mode": False,
        },
    }

    learner = ComponentBuilder.build_meta_learner(
        meta_cfg,
        knowledge_store,
        world_model,
        mock_logger,
        mock_metrics,
        config,
    )

    from polaris.meta_learner.bayesian_optimizer import AcquisitionFunction
    from polaris.meta_learner.statistical import StatisticalMetaLearner

    assert isinstance(learner, StatisticalMetaLearner)
    assert learner.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT
    assert learner.exploration_weight == pytest.approx(0.25)
    assert learner.min_samples_for_optimization == 7


def test_build_meta_learner_llm_failure_returns_none(monkeypatch, mock_logger):
    knowledge_store = InMemoryKnowledgeStore()
    world_model = StatisticalWorldModel(knowledge_store)
    config = _cfg()

    def fake_create_llm_client(*_args, **_kwargs):
        raise RuntimeError("no credentials")

    monkeypatch.setattr(
        "polaris.infrastructure.llm.create_llm_client",
        fake_create_llm_client,
    )

    learner = ComponentBuilder.build_meta_learner(
        {"type": "llm", "llm": {"provider": "openai"}},
        knowledge_store,
        world_model,
        mock_logger,
        None,
        config,
    )

    assert learner is None
    assert any(
        level == "warning" and "Failed to initialize LLM meta-learner" in message
        for level, message, _ in mock_logger.logs
    )


def test_build_connectors_skips_disabled_and_invalid_entries(mock_logger, monkeypatch):
    systems = [
        SimpleNamespace(id="disabled", connector_type="ok", enabled=False),
        SimpleNamespace(id="missing-factory", connector_type="missing", enabled=True),
        SimpleNamespace(id="factory-error", connector_type="boom", enabled=True),
        SimpleNamespace(id="ok", connector_type="ok", enabled=True),
    ]
    config = _cfg(observability={"metrics": {"components": {"connectors": False}}})

    def get_factory(connector_type):
        if connector_type == "missing":
            return None
        if connector_type == "boom":

            def _boom(*_args, **_kwargs):
                raise RuntimeError("boom")

            return _boom

        def _ok(system, _logger, metrics):
            return {"id": system.id, "metrics": metrics}

        return _ok

    monkeypatch.setattr("polaris.core.factories.get_connector_factory", get_factory)
    connectors = ComponentBuilder.build_connectors(systems, mock_logger, object(), config)

    assert connectors == [{"id": "ok", "metrics": None}]
    error_messages = [message for level, message, _ in mock_logger.logs if level == "error"]
    assert any("No connector factory registered" in msg for msg in error_messages)
    assert any("Failed to create connector" in msg for msg in error_messages)


def test_build_metrics_export_config_merges_config_and_cli():
    class ExportingMetrics:
        def export_to_file(self, *_args, **_kwargs):
            return None

    config = _cfg(
        observability={
            "metrics": {
                "export": {
                    "enabled": True,
                    "auto_export_interval_minutes": 10,
                    "output_dir": "./from-config",
                    "formats": ["json"],
                    "experiment_name": "from-config",
                    "include_timestamp": False,
                }
            }
        }
    )
    cli_overrides = {
        "metrics_export_dir": "./from-cli",
        "metrics_auto_export_interval": 5,
        "metrics_export_formats": ["csv"],
        "metrics_experiment_name": "from-cli",
    }

    export_cfg = ComponentBuilder.build_metrics_export_config(
        config,
        cli_overrides,
        ExportingMetrics(),
    )

    assert export_cfg == {
        "enabled": True,
        "interval_minutes": 5,
        "output_dir": "./from-cli",
        "formats": ["csv"],
        "experiment_name": "from-cli",
        "include_timestamp": False,
    }
    assert ComponentBuilder.build_metrics_export_config(config, {}, object()) == {"enabled": False}


def test_should_collect_and_meta_interval_helpers():
    config = _cfg(observability={"metrics": {"components": {"strategy": False}}})

    assert ComponentBuilder.should_collect(config, "unknown", object()) is True
    assert ComponentBuilder.should_collect(config, "strategy", object()) is False
    assert ComponentBuilder.should_collect(config, "strategy", None) is False

    assert ComponentBuilder.resolve_meta_learning_interval({"analysis_interval_hours": 2}) == 7200.0
    assert ComponentBuilder.resolve_meta_learning_interval({"analysis_interval_hours": 0}) == 3600.0
    assert (
        ComponentBuilder.resolve_meta_learning_interval({"analysis_interval_hours": "bad"})
        == 3600.0
    )
    assert ComponentBuilder.resolve_meta_learning_interval(None) == 3600.0
