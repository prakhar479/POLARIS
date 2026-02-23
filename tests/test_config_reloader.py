"""Tests for hot-reload behavior extracted in ConfigReloader."""

from types import SimpleNamespace

import pytest

from polaris.core.config_reloader import ConfigReloader


class _StrategyThresholdReactive:
    async def apply_config_update(self, payload):
        self.payload = payload


_StrategyThresholdReactive.__name__ = "ThresholdReactiveStrategy"


class _StrategyWithError:
    async def apply_config_update(self, payload):
        _ = payload
        raise RuntimeError("cannot apply")


_StrategyWithError.__name__ = "ThresholdReactiveStrategy"


class _Metrics:
    def __init__(self):
        self.calls = []

    def increment(self, metric, value=1.0, tags=None):
        self.calls.append(("increment", metric, value, tags))


def _cfg(core_framework_enabled=True):
    return SimpleNamespace(
        observability={
            "metrics": {
                "components": {
                    "core_framework": core_framework_enabled,
                }
            }
        }
    )


@pytest.mark.asyncio
async def test_maybe_reload_returns_none_without_config_path(mock_logger):
    reloader = ConfigReloader(
        config_path=None,
        strategy=None,
        logger=mock_logger,
        metrics=None,
        config=_cfg(),
    )
    assert await reloader.maybe_reload() is None


@pytest.mark.asyncio
async def test_maybe_reload_skips_when_mtime_unchanged(monkeypatch, mock_logger):
    monkeypatch.setattr("polaris.core.config_reloader.os.path.getmtime", lambda _p: 10.0)
    reloader = ConfigReloader(
        config_path="conf.yaml",
        strategy=None,
        logger=mock_logger,
        metrics=None,
        config=_cfg(),
    )

    assert await reloader.maybe_reload() is None


@pytest.mark.asyncio
async def test_maybe_reload_success_applies_strategy_and_emits_metrics(monkeypatch, mock_logger):
    mtimes = iter([100.0, 101.0])
    monkeypatch.setattr("polaris.core.config_reloader.os.path.getmtime", lambda _p: next(mtimes))

    strategy = _StrategyThresholdReactive()
    metrics = _Metrics()
    initial_cfg = _cfg(core_framework_enabled=True)
    new_strategy_cfg = SimpleNamespace(
        type="threshold",
        threshold={"thresholds": {"cpu": {"high": 90}}},
        llm_reasoning=None,
        hybrid=None,
        agentic_llm=None,
    )
    new_cfg = SimpleNamespace(strategy=new_strategy_cfg, observability=initial_cfg.observability)

    monkeypatch.setattr("polaris.infrastructure.config.load_config", lambda _p: new_cfg)

    reloader = ConfigReloader(
        config_path="conf.yaml",
        strategy=strategy,
        logger=mock_logger,
        metrics=metrics,
        config=initial_cfg,
    )

    loaded = await reloader.maybe_reload()

    assert loaded is new_cfg
    assert strategy.payload == {"thresholds": {"cpu": {"high": 90}}}
    metric_names = [entry[1] for entry in metrics.calls]
    assert "polaris.config.hot_reload.attempts" in metric_names
    assert "polaris.config.hot_reload.success" in metric_names
    assert any(
        level == "info" and "Applied hot-reload from updated configuration" in message
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_maybe_reload_error_path_emits_warning_and_error_metric(monkeypatch, mock_logger):
    mtimes = iter([100.0, 101.0])
    monkeypatch.setattr("polaris.core.config_reloader.os.path.getmtime", lambda _p: next(mtimes))
    monkeypatch.setattr(
        "polaris.infrastructure.config.load_config",
        lambda _p: (_ for _ in ()).throw(ValueError("bad config")),
    )

    metrics = _Metrics()
    reloader = ConfigReloader(
        config_path="conf.yaml",
        strategy=_StrategyThresholdReactive(),
        logger=mock_logger,
        metrics=metrics,
        config=_cfg(core_framework_enabled=True),
    )

    loaded = await reloader.maybe_reload()

    assert loaded is None
    metric_names = [entry[1] for entry in metrics.calls]
    assert "polaris.config.hot_reload.attempts" in metric_names
    assert "polaris.config.hot_reload.errors" in metric_names
    assert any(
        level == "warning" and "Hot-reload skipped due to error: bad config" in message
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_apply_strategy_hot_reload_type_change_requires_restart(mock_logger):
    strategy = _StrategyThresholdReactive()
    reloader = ConfigReloader(
        config_path=None,
        strategy=strategy,
        logger=mock_logger,
        metrics=None,
        config=_cfg(),
    )
    desired = SimpleNamespace(
        type="llm_reasoning",
        threshold=None,
        llm_reasoning={"temperature": 0.2},
        hybrid=None,
        agentic_llm=None,
    )

    await reloader._apply_strategy_hot_reload(desired)

    assert not hasattr(strategy, "payload")
    assert any(
        level == "info" and "restart required to apply" in message
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_apply_strategy_hot_reload_strategy_update_failure_logs_warning(mock_logger):
    reloader = ConfigReloader(
        config_path=None,
        strategy=_StrategyWithError(),
        logger=mock_logger,
        metrics=None,
        config=_cfg(),
    )
    desired = SimpleNamespace(
        type="threshold",
        threshold={"thresholds": {"latency": {"high": 300}}},
        llm_reasoning=None,
        hybrid=None,
        agentic_llm=None,
    )

    await reloader._apply_strategy_hot_reload(desired)

    assert any(
        level == "warning" and "Failed to apply strategy config update: cannot apply" in message
        for level, message, _ in mock_logger.logs
    )


def test_emit_respects_component_metrics_toggle(mock_logger):
    metrics = _Metrics()
    reloader = ConfigReloader(
        config_path=None,
        strategy=None,
        logger=mock_logger,
        metrics=metrics,
        config=_cfg(core_framework_enabled=False),
    )

    reloader._emit("polaris.config.hot_reload.attempts")
    assert metrics.calls == []

    reloader_enabled = ConfigReloader(
        config_path=None,
        strategy=None,
        logger=mock_logger,
        metrics=metrics,
        config=_cfg(core_framework_enabled=True),
    )
    reloader_enabled._emit("polaris.config.hot_reload.attempts")
    assert metrics.calls[-1][1] == "polaris.config.hot_reload.attempts"
