"""Metrics export utilities for Polaris."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from polaris.infrastructure.observability.metrics import SimpleMetricsCollector


def export_polaris_metrics(
    metrics_collector: SimpleMetricsCollector,
    output_dir: Union[str, Path],
    experiment_name: Optional[str] = None,
    formats: Optional[list] = None,
) -> dict:
    """Export Polaris metrics with standardized naming and structure.

    Args:
        metrics_collector: The metrics collector instance
        output_dir: Directory to save metrics files
        experiment_name: Optional experiment identifier for file naming
        formats: List of formats to export ('json', 'csv'). Defaults to both.

    Returns:
        Dict with paths to exported files
    """
    if formats is None:
        formats = ["json", "csv"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create base filename
    if experiment_name:
        base_name = f"polaris_metrics_{experiment_name}_{timestamp}"
    else:
        base_name = f"polaris_metrics_{timestamp}"

    exported_files = {}

    for format_type in formats:
        if format_type.lower() in ["json", "csv"]:
            file_path = output_dir / f"{base_name}.{format_type.lower()}"
            metrics_collector.export_to_file(file_path, format_type)
            exported_files[format_type] = str(file_path)

    return exported_files


def create_metrics_summary_report(
    metrics_collector: SimpleMetricsCollector, output_file: Union[str, Path]
) -> None:
    """Create a human-readable metrics summary report.

    Args:
        metrics_collector: The metrics collector instance
        output_file: Path to save the report
    """
    summary = metrics_collector.get_summary()

    with open(output_file, "w") as f:
        f.write("Polaris Framework Metrics Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Counters section
        f.write("COUNTERS\n")
        f.write("-" * 20 + "\n")
        if summary["counters"]:
            for metric, value in sorted(summary["counters"].items()):
                f.write(f"{metric}: {value}\n")
        else:
            f.write("No counters recorded\n")
        f.write("\n")

        # Gauges section
        f.write("GAUGES\n")
        f.write("-" * 20 + "\n")
        if summary["gauges"]:
            for metric, value in sorted(summary["gauges"].items()):
                f.write(f"{metric}: {value}\n")
        else:
            f.write("No gauges recorded\n")
        f.write("\n")

        # Histograms section
        f.write("HISTOGRAMS\n")
        f.write("-" * 20 + "\n")
        if summary["histograms"]:
            for metric, stats in sorted(summary["histograms"].items()):
                f.write(f"{metric}:\n")
                f.write(f"  Count: {stats['count']}\n")
                f.write(f"  Min: {stats['min']: .2f}\n")
                f.write(f"  Max: {stats['max']: .2f}\n")
                f.write(f"  Avg: {stats['avg']: .2f}\n")
                f.write("\n")
        else:
            f.write("No histograms recorded\n")
