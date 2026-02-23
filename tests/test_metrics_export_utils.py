"""Tests for metrics export utility functions."""

from pathlib import Path

from polaris.infrastructure.observability.export import (
    create_metrics_summary_report,
    export_polaris_metrics,
)


class _Collector:
    def __init__(self, summary=None):
        self.export_calls = []
        self._summary = summary or {
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

    def export_to_file(self, file_path, fmt):
        self.export_calls.append((str(file_path), fmt))

    def get_summary(self):
        return self._summary


def test_export_polaris_metrics_creates_files_for_supported_formats(tmp_path: Path):
    collector = _Collector()
    out = export_polaris_metrics(
        metrics_collector=collector,
        output_dir=tmp_path / "nested" / "metrics",
        experiment_name="exp1",
        formats=["JSON", "csv", "txt"],
    )

    assert set(out.keys()) == {"JSON", "csv"}
    for path in out.values():
        assert str(tmp_path) in path
    assert len(collector.export_calls) == 2
    assert collector.export_calls[0][0].endswith(".json")
    assert collector.export_calls[1][0].endswith(".csv")


def test_export_polaris_metrics_defaults_to_json_and_csv(tmp_path: Path):
    collector = _Collector()
    out = export_polaris_metrics(collector, tmp_path)
    assert set(out.keys()) == {"json", "csv"}


def test_create_metrics_summary_report_with_data(tmp_path: Path):
    collector = _Collector(
        summary={
            "counters": {"polaris.calls": 3},
            "gauges": {"polaris.latency": 12.5},
            "histograms": {
                "polaris.loop_duration": {
                    "count": 2,
                    "min": 0.5,
                    "max": 1.0,
                    "avg": 0.75,
                }
            },
        }
    )
    report_file = tmp_path / "report.txt"

    create_metrics_summary_report(collector, report_file)

    text = report_file.read_text(encoding="utf-8")
    assert "Polaris Framework Metrics Summary" in text
    assert "polaris.calls: 3" in text
    assert "polaris.latency: 12.5" in text
    assert "polaris.loop_duration:" in text
    assert "Count: 2" in text


def test_create_metrics_summary_report_with_empty_sections(tmp_path: Path):
    collector = _Collector()
    report_file = tmp_path / "empty.txt"

    create_metrics_summary_report(collector, report_file)

    text = report_file.read_text(encoding="utf-8")
    assert "No counters recorded" in text
    assert "No gauges recorded" in text
    assert "No histograms recorded" in text
