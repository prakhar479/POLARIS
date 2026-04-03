#!/usr/bin/env python3
"""Advanced metrics export example for Polaris.

This example demonstrates: 1. Comprehensive metrics collection across framework
components 2. Multiple export formats and utilities 3. Experiment-friendly naming and
organization 4. Human-readable summary reports
"""

import asyncio
import tempfile
from pathlib import Path

from polaris.core.polaris import Polaris
from polaris.infrastructure.observability import (
    SimpleMetricsCollector,
    create_metrics_summary_report,
    export_polaris_metrics,
)
from polaris.strategies import ThresholdReactiveStrategy


async def simulate_framework_activity(polaris: Polaris):
    """Simulate typical framework activity to generate metrics."""
    # Simulate monitoring loop metrics
    for i in range(5):
        polaris.metrics.increment("polaris.monitoring.loop_iterations")
        polaris.metrics.histogram("polaris.monitoring.loop_duration_seconds", 2.5 + i * 0.3)
        polaris.metrics.gauge("polaris.monitoring.systems_processed", 3)

        # Simulate telemetry collection
        for system_id in ["web-server", "database", "cache"]:
            polaris.metrics.increment("polaris.telemetry.collected", tags={"system_id": system_id})
            polaris.metrics.increment(
                "polaris.knowledge.state_stored", tags={"system_id": system_id}
            )

    # Simulate strategy assessments and adaptations
    for system_id in ["web-server", "database"]:
        polaris.metrics.increment("polaris.strategy.assessments", tags={"system_id": system_id})

        # Simulate some threshold breaches
        if system_id == "web-server":
            polaris.metrics.increment(
                "polaris.strategy.threshold.high_threshold_exceeded",
                tags={"metric": "cpu_usage", "system_id": system_id},
            )
            polaris.metrics.increment(
                "polaris.adaptations.proposed",
                tags={"system_id": system_id, "action_type": "scale_up"},
            )
            polaris.metrics.increment(
                "polaris.adaptations.executed",
                tags={"system_id": system_id, "action_type": "scale_up", "status": "success"},
            )

    # Simulate event bus activity
    polaris.metrics.increment(
        "polaris.event_bus.events_published", tags={"event_type": "TelemetryEvent"}, value=15
    )
    polaris.metrics.increment(
        "polaris.event_bus.events_published", tags={"event_type": "AdaptationEvent"}, value=2
    )
    polaris.metrics.histogram("polaris.event_bus.handler_duration_seconds", 0.05)
    polaris.metrics.histogram("polaris.event_bus.handler_duration_seconds", 0.12)

    # Simulate registry activity
    polaris.metrics.increment("polaris.registry.connector_registered", value=3)
    polaris.metrics.gauge("polaris.registry.total_connectors", 3)
    polaris.metrics.increment("polaris.registry.connector_accessed", value=25)


async def main():
    """Demonstrate advanced metrics export capabilities."""
    print("Setting up Polaris with comprehensive metrics collection...")

    # Create metrics collector
    metrics = SimpleMetricsCollector()

    # Create strategy with metrics
    strategy = ThresholdReactiveStrategy(
        thresholds={
            "cpu_usage": {"high": 75.0, "low": 25.0},
            "memory_usage": {"high": 80.0, "low": 30.0},
            "response_time_ms": {"high": 500.0},
        },
        cooldown_seconds=60,
        metrics=metrics,
    )

    # Initialize Polaris
    polaris = Polaris(strategy=strategy, metrics=metrics, enable_meta_learning=False)

    # Simulate framework activity
    print("Simulating framework activity...")
    await simulate_framework_activity(polaris)

    # Show current metrics
    summary = polaris.get_metrics_summary()
    print("\nCollected metrics: ")
    print(f"  Counters: {len(summary['counters'])}")
    print(f"  Gauges: {len(summary['gauges'])}")
    print(f"  Histograms: {len(summary['histograms'])}")

    # Export metrics using utility functions
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print(f"\nExporting metrics to: {temp_path}")

        # Export with experiment naming
        exported_files = export_polaris_metrics(
            metrics_collector=metrics,
            output_dir=temp_path,
            experiment_name="demo_run_001",
            formats=["json", "csv"],
        )

        print("Exported files:")
        for format_type, file_path in exported_files.items():
            print(f"  {format_type.upper()}: {file_path}")

            # Show file size
            size = Path(file_path).stat().st_size
            print(f"    Size: {size} bytes")

        # Create human-readable summary report
        report_file = temp_path / "metrics_summary_report.txt"
        create_metrics_summary_report(metrics, report_file)
        print(f"\nSummary report: {report_file}")

        # Show part of the report
        with open(report_file) as f:
            lines = f.readlines()
            print("\nReport preview:")
            for line in lines[:15]:  # Show first 15 lines
                print(f"  {line.rstrip()}")
            if len(lines) > 15:
                print(f"  ... ({len(lines) - 15} more lines)")

        # Demonstrate programmatic access to specific metrics
        print("\nKey performance indicators:")

        # Calculate success rate
        total_adaptations = summary["counters"].get(
            "polaris.adaptations.executed{action_type=scale_up,status=success,system_id=web-server}",
            0,
        )
        if total_adaptations > 0:
            print(f"  Adaptation success rate: 100% ({total_adaptations} successful)")

        # Show average response times
        for hist_name, hist_data in summary["histograms"].items():
            if "duration" in hist_name:
                print(
                    f"  {hist_name}: avg={hist_data['avg'] : .3f}s, max={hist_data['max'] : .3f}s"
                )

        # Show system activity
        total_telemetry = sum(
            v for k, v in summary["counters"].items() if k.startswith("polaris.telemetry.collected")
        )
        print(f"  Total telemetry collections: {total_telemetry}")

    print("\nAdvanced metrics export demonstration complete!")
    print("\nThis example shows how to:")
    print("- Collect metrics across all framework components")
    print("- Export in multiple formats with experiment naming")
    print("- Generate human-readable reports")
    print("- Access metrics programmatically for analysis")


if __name__ == "__main__":
    asyncio.run(main())
