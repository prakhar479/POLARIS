#!/usr/bin/env python3
"""SWIM Experiment Metrics Extraction & Analysis Tool.

Extracts telemetry, adaptation decisions, and performance metrics from either:
1. Structured POLARIS JSON/CSV metrics exports (preferred, high fidelity)
2. POLARIS console session logs (fallback regex parser)

Calculates the canonical SWIM utility score:
    utility = 0.5 * (1.0 / (1.0 + response_time / 1000.0)) + 0.5 * dimmer
and evaluates SLA compliance, server churn, and adaptation behavior.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_structured_metrics_json(json_path: Path | str) -> Dict[str, Any]:
    """Parse structured POLARIS JSON metrics export."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    counters = metrics.get("counters", {})
    gauges = metrics.get("gauges", {})
    histograms = metrics.get("histograms", {})

    # Extract actions from counters
    actions_breakdown: Dict[str, int] = {}
    for key, val in counters.items():
        if "polaris.adaptations.executed" in key:
            match = re.search(r"action_type=([a-zA-Z0-9_]+)", key)
            if match:
                action_name = match.group(1)
                actions_breakdown[action_name] = actions_breakdown.get(action_name, 0) + int(val)

    # Extract tool usage
    tools_used: Dict[str, int] = {}
    for key, val in counters.items():
        if "polaris.tool.execution.success" in key:
            match = re.search(r"tool=([a-zA-Z0-9_]+)", key)
            if match:
                tool_name = match.group(1)
                tools_used[tool_name] = int(val)

    # Server count and dimmer from gauges/histograms
    total_cycles = int(counters.get("polaris.telemetry.collected{system_id=swim}", 0))

    return {
        "source": "structured_json",
        "file": str(json_path),
        "total_cycles": total_cycles,
        "actions_breakdown": actions_breakdown,
        "tools_used": tools_used,
        "gauges": gauges,
        "histograms": histograms,
    }


def parse_log_file(log_file: Path | str) -> Dict[str, Any]:
    """Parse POLARIS log file to extract timeline data."""
    data: Dict[str, List[Any]] = {
        "timestamps": [],
        "response_times": [],
        "utilizations": [],
        "dimmers": [],
        "server_counts": [],
        "actions": [],
    }

    if not os.path.exists(log_file):
        return data

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_dimmer = 0.5
    last_servers = 1

    for line in lines:
        if "Adaptation" not in line and "decision:" not in line and "set_dimmer" not in line:
            continue

        ts_match = re.match(r"(\d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            continue
        timestamp_str = ts_match.group(1)

        try:
            timestamp = datetime.strptime(timestamp_str, "%H:%M:%S")
        except ValueError:
            continue

        action_type = "no_adaptation"
        if "set_dimmer" in line:
            dimmer_match = re.search(r"(?:set_dimmer|command=set_dimmer)\s*=?\s*([\d.]+)", line)
            if dimmer_match:
                last_dimmer = float(dimmer_match.group(1))
            action_type = "set_dimmer"
        elif "scale_up" in line:
            action_type = "scale_up"
            last_servers = min(last_servers + 1, 3)
        elif "scale_down" in line:
            action_type = "scale_down"
            last_servers = max(last_servers - 1, 1)

        response_time: Optional[float] = None
        utilization: Optional[float] = None

        rt_patterns = [
            r"(?:~|≈)\s*([\d.]+)\s*(?:ms|milliseconds)",
            r"([\d.]+)\s*ms",
            r"response_time[^\d.]*([\d.]+)",
        ]
        for pattern in rt_patterns:
            rt_match = re.search(pattern, line, re.IGNORECASE)
            if rt_match:
                val = float(rt_match.group(1))
                if 10.0 < val < 10000.0:
                    response_time = val
                    break

        util_match = re.search(r"utilization[^\d.]*([\d.]+)", line, re.IGNORECASE)
        if util_match:
            val = float(util_match.group(1))
            if 0.0 <= val <= 1.0:
                utilization = val

        servers_match = re.search(r"(\d+)\s*servers?", line, re.IGNORECASE)
        if servers_match:
            last_servers = int(servers_match.group(1))

        data["timestamps"].append(timestamp)
        data["response_times"].append(response_time)
        data["utilizations"].append(utilization)
        data["dimmers"].append(last_dimmer)
        data["server_counts"].append(last_servers)
        data["actions"].append(action_type)

    return data


def compute_swim_utility(response_time: float, dimmer: float) -> float:
    """Calculate utility score for given response time and dimmer."""
    rt_component = 1.0 / (1.0 + (response_time / 1000.0))
    return 0.5 * rt_component + 0.5 * dimmer


def create_analysis_plots(
    data: Dict[str, List[Any]],
    sla_target: float,
    output_file: Path | str,
) -> Dict[str, Any]:
    """Generate 3-panel visualization and compute summary metrics."""
    timestamps = data.get("timestamps", [])
    response_times = [v for v in data.get("response_times", []) if v is not None]
    dimmers = data.get("dimmers", [])
    utilizations = [v for v in data.get("utilizations", []) if v is not None]
    actions = data.get("actions", [])

    utilities: List[float] = []
    for rt, dm in zip(data.get("response_times", []), dimmers):
        if rt is not None and dm is not None:
            utilities.append(compute_swim_utility(rt, dm))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Response Time & SLA
    ax1 = axes[0]
    indices = list(range(len(timestamps))) if timestamps else []
    if response_times:
        rt_indices = [i for i, v in enumerate(data.get("response_times", [])) if v is not None]
        ax1.plot(rt_indices, response_times, "b-o", label="Response Time (ms)", markersize=4)
        ax1.axhline(
            y=sla_target, color="r", linestyle="--", label=f"SLA Target ({sla_target:.0f} ms)"
        )
        ax1.set_ylabel("Response Time (ms)")
        ax1.set_title("SWIM Self-Adaptation Experiment Performance", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

    # 2. Dimmer & Utility
    ax2 = axes[1]
    if dimmers:
        ax2.plot(indices, dimmers, "g--s", label="Dimmer (Optional Content)", markersize=4)
    if utilities:
        u_indices = [
            i
            for i, (rt, dm) in enumerate(zip(data.get("response_times", []), dimmers))
            if rt is not None and dm is not None
        ]
        ax2.plot(u_indices, utilities, "m-^", label="Utility Score", markersize=4)
    ax2.set_ylabel("Score / Ratio [0, 1]")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Adaptation Actions
    ax3 = axes[2]
    action_colors = {
        "scale_up": "red",
        "scale_down": "darkorange",
        "set_dimmer": "blue",
        "no_adaptation": "lightgray",
    }
    for i, action in enumerate(actions):
        color = action_colors.get(action, "black")
        if action != "no_adaptation":
            ax3.scatter(i, 1, c=color, s=80, zorder=3)
            ax3.annotate(action, (i, 1.05), rotation=45, ha="right", fontsize=8)

    ax3.set_ylabel("Actions")
    ax3.set_xlabel("Adaptation Cycle / Timestep")
    ax3.set_ylim(0.5, 1.5)
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()

    # Calculate summary
    valid_rts = [v for v in response_times if not np.isnan(v)]
    sla_violations = sum(1 for v in valid_rts if v > sla_target)
    sla_violation_rate = (sla_violations / len(valid_rts)) if valid_rts else 0.0

    return {
        "total_data_points": len(timestamps),
        "avg_response_time_ms": float(np.mean(valid_rts)) if valid_rts else None,
        "p95_response_time_ms": float(np.percentile(valid_rts, 95)) if valid_rts else None,
        "sla_target_ms": sla_target,
        "sla_violations": sla_violations,
        "sla_violation_rate": sla_violation_rate,
        "avg_utilization": float(np.mean(utilizations)) if utilizations else None,
        "final_dimmer": dimmers[-1] if dimmers else None,
        "avg_utility": float(np.mean(utilities)) if utilities else None,
        "final_utility": utilities[-1] if utilities else None,
        "total_adaptations": sum(1 for a in actions if a != "no_adaptation"),
    }


def find_latest_file(pattern: str) -> Optional[Path]:
    """Find the most recently modified file matching pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return Path(latest)


def main() -> None:
    """Execute SWIM metrics extraction and generate evaluation report."""
    parser = argparse.ArgumentParser(description="SWIM Metrics Extraction & Evaluation Tool")
    parser.add_argument(
        "--metrics-file", "-m", type=str, help="Path to POLARIS metrics JSON or CSV"
    )
    parser.add_argument("--log-file", "-l", type=str, help="Path to POLARIS run log file")
    parser.add_argument(
        "--output-plot",
        "-o",
        type=str,
        default="swim_metrics_plot.png",
        help="Path for generated metrics plot",
    )
    parser.add_argument(
        "--output-summary",
        "-s",
        type=str,
        default="swim_experiment_summary.json",
        help="Path for generated summary JSON",
    )
    parser.add_argument(
        "--sla-target",
        type=float,
        default=750.0,
        help="Response time SLA threshold in ms (default: 750.0)",
    )

    args = parser.parse_args()

    # Resolve input sources
    repo_root = Path(__file__).resolve().parents[1]
    log_file: Optional[Path] = Path(args.log_file) if args.log_file else None
    metrics_file: Optional[Path] = Path(args.metrics_file) if args.metrics_file else None

    if not metrics_file:
        metrics_file = find_latest_file(str(repo_root / "metrics" / "swim" / "*.json"))
        if not metrics_file:
            metrics_file = find_latest_file(str(repo_root / "metrics" / "*.json"))

    if not log_file:
        log_file = find_latest_file(str(repo_root / "logs" / "swim_polaris_run_*.log"))
        if not log_file:
            log_file = find_latest_file(str(repo_root / "logs" / "*.log"))

    print("📊 SWIM Experiment Metrics Extractor")
    print("====================================")

    structured_summary: Dict[str, Any] = {}
    if metrics_file and metrics_file.exists():
        print(f"✓ Found structured metrics: {metrics_file.name}")
        structured_summary = parse_structured_metrics_json(metrics_file)

    timeline_data: Dict[str, Any] = {}
    if log_file and log_file.exists():
        print(f"✓ Found session log: {log_file.name}")
        timeline_data = parse_log_file(log_file)

    if not structured_summary and not timeline_data.get("timestamps"):
        print("⚠️  No metrics or log files found to analyze.")
        print(f"   Looked in: {repo_root}/metrics/ and {repo_root}/logs/")
        return

    # Generate analysis & plots
    plot_path = Path(args.output_plot).resolve()
    summary = create_analysis_plots(timeline_data, args.sla_target, plot_path)

    # Merge structured metrics if available
    if structured_summary:
        summary["structured_metrics"] = structured_summary

    # Output report
    print("\n" + "=" * 50)
    print("SWIM EXPERIMENT EVALUATION REPORT")
    print("=" * 50)
    if summary.get("avg_response_time_ms") is not None:
        print(f"Avg Response Time : {summary['avg_response_time_ms']:.2f} ms")
        print(f"P95 Response Time : {summary['p95_response_time_ms']:.2f} ms")
        print(f"SLA Target        : {summary['sla_target_ms']:.1f} ms")
        print(
            f"SLA Violation Rate: {summary['sla_violation_rate'] * 100:.1f}% ({summary['sla_violations']} breaches)"
        )
    if summary.get("avg_utility") is not None:
        print(f"Mean Utility Score: {summary['avg_utility']:.4f}")
        print(f"Final Utility     : {summary['final_utility']:.4f}")
    if summary.get("final_dimmer") is not None:
        print(f"Final Dimmer      : {summary['final_dimmer']:.2f}")
    print(f"Total Adaptations : {summary['total_adaptations']}")

    if structured_summary.get("actions_breakdown"):
        print("\nAction Breakdown:")
        for action, count in structured_summary["actions_breakdown"].items():
            print(f"  - {action}: {count}")

    if structured_summary.get("tools_used"):
        print("\nTool Usage Breakdown:")
        for tool, count in structured_summary["tools_used"].items():
            print(f"  - {tool}: {count}")
    print("=" * 50)

    # Save summary
    summary_path = Path(args.output_summary).resolve()
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved plot   : {plot_path}")
    print(f"✓ Saved summary: {summary_path}\n")


if __name__ == "__main__":
    main()
