"""Tests for meta-learning and metrics-export background loops."""

import asyncio
from types import SimpleNamespace

import pytest

from polaris.core.meta_learning_loop import MetaLearningLoop
from polaris.core.metrics_export_loop import MetricsExportLoop


class _Metrics:
    def __init__(self):
        self.calls = []

    def increment(self, metric, value=1.0, tags=None):
        self.calls.append(("increment", metric, value, tags))

    def gauge(self, metric, value, tags=None):
        self.calls.append(("gauge", metric, value, tags))

    def histogram(self, metric, value, tags=None):
        self.calls.append(("histogram", metric, value, tags))

    def export_to_file(self, _path, _fmt):
        return None


class _Registry:
    def __init__(self, systems):
        self._systems = list(systems)

    def system_ids(self):
        return list(self._systems)


class _MetaLearner:
    def __init__(self, *, proposals=None, validated=None, applied=None, fail=False):
        self.proposals = proposals if proposals is not None else ["p1", "p2"]
        self.validated = validated if validated is not None else ["p1"]
        self.applied = applied if applied is not None else ["p1"]
        self.fail = fail
        self.calls = []

    async def analyze_performance(self, system_id):
        self.calls.append(("analyze", system_id))
        if self.fail:
            raise RuntimeError("analysis failed")
        return {"system_id": system_id}

    async def propose_strategy_updates(self, strategy, analysis):
        self.calls.append(("propose", strategy, analysis))
        return list(self.proposals)

    async def validate_proposals(self, proposals):
        self.calls.append(("validate", list(proposals)))
        return list(self.validated)

    async def apply_proposals(self, strategy, proposals):
        self.calls.append(("apply", strategy, list(proposals)))
        return list(self.applied)


def _cfg_meta(enabled=True):
    return SimpleNamespace(
        observability={
            "metrics": {
                "components": {
                    "meta_learner": enabled,
                    "core_framework": enabled,
                }
            }
        }
    )


@pytest.mark.asyncio
async def test_meta_learning_run_for_system_success(mock_logger):
    metrics = _Metrics()
    learner = _MetaLearner(proposals=["a", "b"], validated=["a"], applied=["a"])
    loop = MetaLearningLoop(
        meta_learner=learner,
        strategy=object(),
        registry=_Registry(["sys-1"]),
        logger=mock_logger,
        metrics=metrics,
        interval_seconds=1.0,
        config=_cfg_meta(enabled=True),
    )

    await loop._run_for_system("sys-1")

    call_metrics = [(kind, name) for kind, name, *_ in metrics.calls]
    assert ("increment", "polaris.meta_learning.analysis_completed") in call_metrics
    assert ("gauge", "polaris.meta_learning.proposals_generated") in call_metrics
    assert ("gauge", "polaris.meta_learning.proposals_validated") in call_metrics
    assert ("gauge", "polaris.meta_learning.proposals_applied") in call_metrics
    assert any(
        level == "info" and "Meta-learner applied 1 parameter updates" in message
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_meta_learning_run_for_system_no_proposals_short_circuits(mock_logger):
    metrics = _Metrics()
    learner = _MetaLearner(proposals=[], validated=["unused"], applied=["unused"])
    loop = MetaLearningLoop(
        meta_learner=learner,
        strategy=object(),
        registry=_Registry(["sys-1"]),
        logger=mock_logger,
        metrics=metrics,
        interval_seconds=1.0,
        config=_cfg_meta(enabled=True),
    )

    await loop._run_for_system("sys-1")

    assert not any(call[0] == "validate" for call in learner.calls)
    assert not any(call[0] == "apply" for call in learner.calls)


@pytest.mark.asyncio
async def test_meta_learning_run_for_system_error_path_emits_error_metric(mock_logger):
    metrics = _Metrics()
    learner = _MetaLearner(fail=True)
    loop = MetaLearningLoop(
        meta_learner=learner,
        strategy=object(),
        registry=_Registry(["sys-err"]),
        logger=mock_logger,
        metrics=metrics,
        interval_seconds=1.0,
        config=_cfg_meta(enabled=True),
    )

    await loop._run_for_system("sys-err")

    assert any(
        kind == "increment"
        and metric == "polaris.meta_learning.errors"
        and tags == {"system_id": "sys-err"}
        for kind, metric, _value, tags in metrics.calls
    )
    assert any(
        level == "error" and "Error in meta-learning for sys-err: analysis failed" in message
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_meta_learning_run_start_stop_and_jitter(monkeypatch, mock_logger):
    metrics = _Metrics()
    learner = _MetaLearner(proposals=[])
    loop = MetaLearningLoop(
        meta_learner=learner,
        strategy=object(),
        registry=_Registry(["sys-1", "sys-2"]),
        logger=mock_logger,
        metrics=metrics,
        interval_seconds=10.0,
        config=_cfg_meta(enabled=True),
    )

    sleep_values = []

    async def fake_sleep(seconds):
        sleep_values.append(seconds)
        loop._running = False

    called_systems = []

    async def fake_run_for_system(system_id):
        called_systems.append(system_id)

    monkeypatch.setattr("polaris.core.meta_learning_loop.random.random", lambda: 0.5)
    monkeypatch.setattr("polaris.core.meta_learning_loop.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(loop, "_run_for_system", fake_run_for_system)

    await loop.run()

    assert sleep_values == [10.0]
    assert called_systems == ["sys-1", "sys-2"]
    metric_names = [metric for kind, metric, *_ in metrics.calls if kind == "increment"]
    assert "polaris.meta_learning.started" in metric_names
    assert "polaris.meta_learning.stopped" in metric_names


@pytest.mark.asyncio
async def test_meta_learning_run_handles_cancelled_sleep(monkeypatch, mock_logger):
    loop = MetaLearningLoop(
        meta_learner=_MetaLearner(),
        strategy=object(),
        registry=_Registry(["sys-1"]),
        logger=mock_logger,
        metrics=_Metrics(),
        interval_seconds=1.0,
        config=_cfg_meta(enabled=True),
    )

    async def cancelled_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr("polaris.core.meta_learning_loop.asyncio.sleep", cancelled_sleep)

    await loop.run()
    assert any(
        level == "info" and "Meta-learning loop stopped" in message
        for level, message, _ in mock_logger.logs
    )


def test_meta_learning_metrics_respect_component_toggle(mock_logger):
    metrics = _Metrics()
    loop = MetaLearningLoop(
        meta_learner=_MetaLearner(),
        strategy=object(),
        registry=_Registry([]),
        logger=mock_logger,
        metrics=metrics,
        interval_seconds=1.0,
        config=_cfg_meta(enabled=False),
    )

    loop._emit("x")
    loop._emit_tagged("y", "sys-1")
    loop._gauge_tagged("z", 1.0, "sys-1")
    assert metrics.calls == []


@pytest.mark.asyncio
async def test_metrics_export_loop_run_happy_path(monkeypatch, mock_logger):
    metrics = _Metrics()
    loop = MetricsExportLoop(
        metrics=metrics,
        export_config={
            "enabled": True,
            "interval_minutes": 1,
            "output_dir": "/tmp/unused",
            "formats": ["json"],
        },
        logger=mock_logger,
        config=_cfg_meta(enabled=True),
    )

    async def fake_sleep(_seconds):
        return None

    async def fake_do_export():
        loop._running = False

    monkeypatch.setattr("polaris.core.metrics_export_loop.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(loop, "_do_export", fake_do_export)

    await loop.run()

    assert any(
        kind == "increment" and metric == "polaris.metrics.auto_export_started"
        for kind, metric, _value, _tags in metrics.calls
    )
    assert any(
        level == "info" and "Metrics auto-export loop stopped" in message
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_metrics_export_loop_early_return_conditions(mock_logger):
    await MetricsExportLoop(
        metrics=None,
        export_config={"enabled": True, "interval_minutes": 1},
        logger=mock_logger,
        config=_cfg_meta(),
    ).run()

    class _NoExporter:
        pass

    await MetricsExportLoop(
        metrics=_NoExporter(),
        export_config={"enabled": True, "interval_minutes": 1},
        logger=mock_logger,
        config=_cfg_meta(),
    ).run()

    await MetricsExportLoop(
        metrics=_Metrics(),
        export_config={"enabled": False, "interval_minutes": 1},
        logger=mock_logger,
        config=_cfg_meta(),
    ).run()

    assert all(
        not (level == "info" and "Starting metrics auto-export" in message)
        for level, message, _ in mock_logger.logs
    )


@pytest.mark.asyncio
async def test_metrics_export_loop_handles_loop_exception(monkeypatch, mock_logger):
    metrics = _Metrics()
    loop = MetricsExportLoop(
        metrics=metrics,
        export_config={
            "enabled": True,
            "interval_minutes": 1,
            "output_dir": "/tmp/unused",
            "formats": ["json"],
        },
        logger=mock_logger,
        config=_cfg_meta(enabled=True),
    )

    async def bad_sleep(_seconds):
        loop._running = False
        raise RuntimeError("sleep failed")

    monkeypatch.setattr("polaris.core.metrics_export_loop.asyncio.sleep", bad_sleep)

    await loop.run()

    assert any(
        kind == "increment" and metric == "polaris.metrics.export_loop_errors"
        for kind, metric, _value, _tags in metrics.calls
    )


@pytest.mark.asyncio
async def test_metrics_export_do_export_success_and_error(monkeypatch, mock_logger):
    metrics = _Metrics()
    loop = MetricsExportLoop(
        metrics=metrics,
        export_config={
            "enabled": True,
            "interval_minutes": 1,
            "output_dir": "/tmp/unused",
            "formats": ["json"],
            "experiment_name": "exp",
        },
        logger=mock_logger,
        config=_cfg_meta(enabled=True),
    )

    monkeypatch.setattr(
        "polaris.infrastructure.observability.export.export_polaris_metrics",
        lambda **_kwargs: {"json": "/tmp/file.json"},
    )
    await loop._do_export()
    assert any(
        kind == "increment" and metric == "polaris.metrics.auto_exports_completed"
        for kind, metric, _value, _tags in metrics.calls
    )
    assert any(kind == "histogram" for kind, *_ in metrics.calls)

    def _boom(**_kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(
        "polaris.infrastructure.observability.export.export_polaris_metrics",
        _boom,
    )
    await loop._do_export()
    assert any(
        kind == "increment" and metric == "polaris.metrics.auto_export_errors"
        for kind, metric, _value, _tags in metrics.calls
    )
