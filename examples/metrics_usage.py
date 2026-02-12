#!/usr/bin/env python3
"""
Example demonstrating metrics collection and export in Polaris.

This example shows how to:
1. Use the framework with metrics collection enabled
2. Export metrics to JSON and CSV formats
3. Access metrics programmatically
"""

import asyncio
import tempfile
from pathlib import Path

from polaris.core.polaris import Polaris
from polaris.infrastructure.observability import SimpleMetricsCollector
from polaris.strategies import ThresholdReactiveStrategy


async def main():
    """Demonstrate metrics collection and export."""
    # Create a custom metrics collector
    metrics = SimpleMetricsCollector()

    # Create a strategy with metrics enabled
    strategy = ThresholdReactiveStrategy(
        thresholds={
            "cpu_usage": {"high": 80.0, "low": 20.0},
            "memory_usage": {"high": 85.0, "low": 25.0},
        },
        cooldown_seconds=30,
        metrics=metrics,
    )

    # Initialize Polaris with custom metrics
    polaris = Polaris(strategy=strategy, metrics=metrics, enable_meta_learning=False)

    print("Starting Polaris with metrics collection...")

    # Simulate some framework activity by manually adding metrics
    # (In real usage, these would be collected automatically)
    metrics.increment("polaris.demo.startup")
    metrics.gauge("polaris.demo.systems_configured", 2)
    metrics.histogram("polaris.demo.response_time_ms", 150.5)
    metrics.histogram("polaris.demo.response_time_ms", 200.2)
    metrics.histogram("polaris.demo.response_time_ms", 175.8)

    # Add some strategy-specific metrics
    metrics.increment("polaris.strategy.threshold.assessments", tags={"system_id": "demo-system"})
    metrics.increment(
        "polaris.strategy.threshold.high_threshold_exceeded",
        tags={"metric": "cpu_usage", "system_id": "demo-system"},
    )

    # Get current metrics summary
    print("\nCurrent metrics summary:")
    summary = polaris.get_metrics_summary()
    print(f"Counters: {len(summary['counters'])}")
    print(f"Gauges: {len(summary['gauges'])}")
    print(f"Histograms: {len(summary['histograms'])}")

    # Export metrics to different formats
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Export as JSON
        json_file = temp_path / "polaris_metrics.json"
        polaris.export_metrics(str(json_file), format="json")
        print(f"\nMetrics exported to JSON: {json_file}")

        # Show JSON content
        with open(json_file) as f:
            import json

            data = json.load(f)
            print("JSON export contains:")
            print(f"  - Export timestamp: {data['export_timestamp']}")
            print(f"  - Counters: {list(data['metrics']['counters'].keys())}")
            print(f"  - Gauges: {list(data['metrics']['gauges'].keys())}")
            print(f"  - Histograms: {list(data['metrics']['histograms'].keys())}")

        # Export as CSV
        csv_file = temp_path / "polaris_metrics.csv"
        polaris.export_metrics(str(csv_file), format="csv")
        print(f"\nMetrics exported to CSV: {csv_file}")

        # Show CSV content
        with open(csv_file) as f:
            lines = f.readlines()
            print(f"CSV export contains {len(lines)} rows: ")
            print("First few rows: ")
            for line in lines[:5]:
                print(f"  {line.strip()}")

    print("\nMetrics integration demonstration complete!")
    print("\nIn a real deployment, metrics would be collected automatically as:")
    print("- Systems are monitored")
    print("- Adaptations are executed")
    print("- Events are published")
    print("- Strategy decisions are made")


if __name__ == "__main__":
    asyncio.run(main())
