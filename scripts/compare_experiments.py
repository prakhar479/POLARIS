#!/usr/bin/env python3
"""Comparative Benchmark & Experiment Evaluation Script.

Ingests experiment summaries (JSON) from multiple POLARIS runs and generates:
1. Formatted comparative console table & Markdown summary
2. Side-by-side comparative visualization plots across:
   - SLA Violation Rate
   - Mean Response Time vs SLA Target
   - Mean Utility Score
   - Total Adaptation Count

Usage:
    python scripts/compare_experiments.py summary1.json summary2.json --labels "Threshold" "THREAD"
    python scripts/compare_experiments.py --dir results/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_summary(file_path: Path | str) -> Optional[Dict[str, Any]]:
    """Load and validate an experiment summary JSON file."""
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_file_name"] = path.stem
        return data
    except Exception as exc:
        print(f"⚠️  Could not read {path}: {exc}")
        return None


def generate_comparison_table(experiments: List[Dict[str, Any]], labels: List[str]) -> str:
    """Generate Markdown comparison table."""
    headers = [
        "Experiment",
        "Data Points",
        "Mean RT (ms)",
        "P95 RT (ms)",
        "SLA Violations",
        "SLA Violation %",
        "Mean Utility",
        "Total Adaptations",
    ]

    rows: List[List[str]] = []
    for exp, label in zip(experiments, labels):
        mean_rt = (
            f"{exp.get('avg_response_time_ms', 0.0):.1f}"
            if exp.get("avg_response_time_ms")
            else "N/A"
        )
        p95_rt = (
            f"{exp.get('p95_response_time_ms', 0.0):.1f}"
            if exp.get("p95_response_time_ms")
            else "N/A"
        )
        violations = str(exp.get("sla_violations", 0))
        violation_rate = f"{exp.get('sla_violation_rate', 0.0) * 100:.1f}%"
        mean_utility = f"{exp.get('avg_utility', 0.0):.4f}" if exp.get("avg_utility") else "N/A"
        adaptations = str(exp.get("total_adaptations", 0))
        pts = str(exp.get("total_data_points", 0))

        rows.append(
            [label, pts, mean_rt, p95_rt, violations, violation_rate, mean_utility, adaptations]
        )

    # Build markdown table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + data_lines)


def plot_comparison_metrics(
    experiments: List[Dict[str, Any]],
    labels: List[str],
    output_path: Path | str,
) -> None:
    """Plot multi-metric comparative bar chart."""
    n_exp = len(experiments)
    if n_exp == 0:
        return

    x = np.arange(n_exp)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean Response Time
    ax1 = axes[0, 0]
    rts = [e.get("avg_response_time_ms", 0.0) or 0.0 for e in experiments]
    bars1 = ax1.bar(x, rts, color="steelblue", width=0.5)
    sla_target = experiments[0].get("sla_target_ms", 750.0)
    ax1.axhline(
        y=sla_target, color="red", linestyle="--", label=f"SLA Target ({sla_target:.0f} ms)"
    )
    ax1.set_ylabel("Response Time (ms)")
    ax1.set_title("Mean Response Time (Lower is Better)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 10,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. SLA Violation Rate (%)
    ax2 = axes[0, 1]
    viol_rates = [(e.get("sla_violation_rate", 0.0) or 0.0) * 100 for e in experiments]
    bars2 = ax2.bar(x, viol_rates, color="crimson", width=0.5)
    ax2.set_ylabel("Violation Rate (%)")
    ax2.set_title("SLA Violation Rate (Lower is Better)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Mean Utility Score
    ax3 = axes[1, 0]
    utilities = [e.get("avg_utility", 0.0) or 0.0 for e in experiments]
    bars3 = ax3.bar(x, utilities, color="seagreen", width=0.5)
    ax3.set_ylabel("Utility Score [0, 1]")
    ax3.set_title("Mean Utility (Higher is Better)", fontweight="bold")
    ax3.set_ylim(0, 1.1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.02,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Total Adaptations Executed
    ax4 = axes[1, 1]
    adaptations = [e.get("total_adaptations", 0) for e in experiments]
    bars4 = ax4.bar(x, adaptations, color="darkorange", width=0.5)
    ax4.set_ylabel("Action Count")
    ax4.set_title("Total Adaptation Actions", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha="right")
    ax4.grid(axis="y", alpha=0.3)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.2,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle("POLARIS Strategy Comparison Evaluation", fontsize=16, fontweight="bold")
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.35, wspace=0.25)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Execute comparative analysis on input experiment summaries."""
    parser = argparse.ArgumentParser(description="Compare multiple POLARIS experiment summaries")
    parser.add_argument("files", nargs="*", help="Paths to experiment summary JSON files")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing summary JSON files")
    parser.add_argument("--labels", "-l", nargs="+", help="Custom labels for each experiment")
    parser.add_argument(
        "--output-plot",
        "-o",
        type=str,
        default="comparison_metrics_plot.png",
        help="Destination path for comparison plot",
    )
    parser.add_argument(
        "--output-table",
        "-t",
        type=str,
        default="comparison_summary.md",
        help="Destination path for markdown comparison table",
    )

    args = parser.parse_args()

    files: List[Path] = [Path(f) for f in args.files]
    if args.dir:
        dir_path = Path(args.dir)
        files.extend(sorted(dir_path.glob("*summary*.json")))

    if not files:
        # Default fallback to find summaries in root
        files = [Path("swim_experiment_summary.json")]

    experiments: List[Dict[str, Any]] = []
    labels: List[str] = []

    for idx, f in enumerate(files):
        data = load_experiment_summary(f)
        if data:
            experiments.append(data)
            if args.labels and idx < len(args.labels):
                labels.append(args.labels[idx])
            else:
                labels.append(data.get("_file_name", f"Run {idx + 1}"))

    if not experiments:
        print("⚠️  No valid experiment summaries found to compare.")
        return

    print("📊 POLARIS Comparative Experiment Evaluation")
    print("============================================")
    table = generate_comparison_table(experiments, labels)
    print("\n" + table + "\n")

    # Save table to Markdown
    table_path = Path(args.output_table).resolve()
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# POLARIS Experiment Comparison\n\n" + table + "\n")
    print(f"✓ Saved markdown table: {table_path}")

    # Plot
    plot_path = Path(args.output_plot).resolve()
    plot_comparison_metrics(experiments, labels, plot_path)
    print(f"✓ Saved comparison plot: {plot_path}\n")


if __name__ == "__main__":
    main()
