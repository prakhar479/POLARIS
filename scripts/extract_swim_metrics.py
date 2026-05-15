#!/usr/bin/env python3
"""
Extract SWIM metrics from POLARIS logs and generate a plot.
SWIM utility is calculated as: utility = 0.5 * (1 / (1 + response_time/1000)) + 0.5 * dimmer
"""

import glob
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_file: str) -> Dict[str, List]:
    """Parse POLARIS log file to extract SWIM metrics."""
    data: Dict[str, List] = {
        "timestamps": [],
        "response_times": [],
        "utilizations": [],
        "dimmers": [],
        "server_counts": [],
        "actions": [],
    }

    with open(log_file) as f:
        lines = f.readlines()

    # Track last known dimmer value
    last_dimmer = 0.5  # Default starting value

    for i, line in enumerate(lines):
        # Look for adaptation or agentic decision lines
        if "Adaptation" not in line and "decision:" not in line:
            continue

        # Extract timestamp
        ts_match = re.match(r"(\d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            continue
        timestamp_str = ts_match.group(1)

        try:
            timestamp = datetime.strptime(timestamp_str, "%H:%M:%S")
        except:
            continue

        # Extract metrics from this line and nearby lines
        reasoning = line

        # Action type
        action_type = ""
        if "set_dimmer" in line and "executed successfully" in line:
            dimmer_match = re.search(r"command=set_dimmer\s+([\d.]+)", line)
            if dimmer_match:
                last_dimmer = float(dimmer_match.group(1))
                action_type = "set_dimmer"
        elif "scale_up" in line:
            action_type = "scale_up"
        elif "scale_down" in line:
            action_type = "scale_down"
        elif "no adaptation" in line.lower() or "no_adaptation" in line:
            action_type = "no_adaptation"

        # Extract metrics from reasoning text
        response_time = None
        utilization = None
        servers = None

        # Response time patterns - look for various formats
        rt_patterns = [
            r"(?:~|≈)\s*([\d.]+)\s*(?:ms|milliseconds)",
            r"(?:~|≈)([\d.]+)",
            r"([\d.]+)\s*ms",
        ]
        for pattern in rt_patterns:
            rt_match = re.search(pattern, reasoning, re.IGNORECASE)
            if rt_match:
                val = float(rt_match.group(1))
                if 10 < val < 5000:  # Response time in ms range
                    response_time = val
                    break

        # Utilization - look for values after "utilization"
        util_match = re.search(r"utilization[^\d.]*([\d.]+)", reasoning, re.IGNORECASE)
        if util_match:
            val = float(util_match.group(1))
            if 0.1 < val < 1.0:  # Valid utilization
                utilization = val

        # Servers
        servers_match = re.search(r"(\d+)\s*servers?", reasoning, re.IGNORECASE)
        if servers_match:
            servers = int(servers_match.group(1))

        # Add data point
        data["timestamps"].append(timestamp)
        data["response_times"].append(response_time)
        data["utilizations"].append(utilization)
        data["dimmers"].append(last_dimmer)
        data["server_counts"].append(servers)
        data["actions"].append(action_type if action_type else "no_adaptation")

    return data


def calculate_utility(
    response_time_ms: Optional[float], dimmer: Optional[float]
) -> Optional[float]:
    """Calculate SWIM utility based on response time and dimmer."""
    if response_time_ms is None or dimmer is None:
        return None
    # Utility = 0.5 * (1 / (1 + RT/1000)) + 0.5 * dimmer
    utility = 0.5 * (1 / (1 + response_time_ms / 1000)) + 0.5 * dimmer
    return utility


def create_plot(
    data: Dict[str, List], output_file: str = "swim_metrics_plot.png"
) -> Dict[str, Optional[float]]:
    """Create a plot of SWIM metrics over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    timestamps = np.arange(len(data["timestamps"]))

    # Calculate utility values
    utilities = []
    for rt, dim in zip(data["response_times"], data["dimmers"]):
        utilities.append(calculate_utility(rt, dim))

    # Plot 1: Response Time and Utilization
    ax1 = axes[0]
    if any(v is not None for v in data["response_times"]):
        ax1.plot(
            timestamps, data["response_times"], "b-o", label="Response Time (ms)", markersize=6
        )
        ax1.axhline(y=750, color="r", linestyle="--", label="SLA Threshold (750ms)")
    ax1.set_ylabel("Response Time (ms)")
    ax1.set_title("SWIM Metrics Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dimmer and Utility
    ax2 = axes[1]
    if any(v is not None for v in data["dimmers"]):
        ax2.plot(timestamps, data["dimmers"], "g-s", label="Dimmer Value", markersize=6)
    if any(v is not None for v in utilities):
        ax2.plot(timestamps, utilities, "r-^", label="Utility", markersize=6)
    ax2.set_ylabel("Value (0-1)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)

    # Plot 3: Actions
    ax3 = axes[2]
    action_colors = {
        "scale_up": "red",
        "scale_down": "orange",
        "set_dimmer": "blue",
        "no_adaptation": "gray",
        "unknown": "black",
    }

    for i, action in enumerate(data["actions"]):
        color = action_colors.get(action, "black")
        ax3.scatter(i, 1, c=color, marker="o", s=100)
        ax3.annotate(action, (i, 1), rotation=45, ha="right", va="bottom", fontsize=8)

    ax3.set_ylabel("Adaptation Actions")
    ax3.set_xlabel("Time Step")
    ax3.set_yticks([])
    ax3.set_xlim(-0.5, len(timestamps) - 0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    # Return summary
    return {
        "avg_response_time": np.mean([v for v in data["response_times"] if v is not None]),
        "avg_utilization": np.mean([v for v in data["utilizations"] if v is not None]),
        "final_dimmer": data["dimmers"][-1] if data["dimmers"] else None,
        "final_utility": utilities[-1] if utilities else None,
        "avg_utility": np.mean([v for v in utilities if v is not None]),
    }


def main() -> None:
    # Find the most recent log file
    log_files = glob.glob("/home/prakhar/dev/prakhar479/polaris/logs/swim_polaris_run_*.log")
    if not log_files:
        print("No log files found!")
        return

    latest_log = max(log_files)
    print(f"Processing: {latest_log}")

    # Parse metrics
    data = parse_log_file(latest_log)
    print(f"\nFound {len(data['timestamps'])} data points")

    # Create plot and get summary
    summary = create_plot(data)

    # Print summary
    print("\n" + "=" * 50)
    print("SWIM EXPERIMENT SUMMARY")
    print("=" * 50)
    avg_rt = summary["avg_response_time"]
    print(
        f"Average Response Time: {avg_rt:.2f} ms"
        if avg_rt is not None and not np.isnan(avg_rt)
        else "Average Response Time: N/A"
    )
    avg_util = summary["avg_utilization"]
    print(
        f"Average Utilization: {avg_util:.2f}"
        if avg_util is not None and not np.isnan(avg_util)
        else "Average Utilization: N/A"
    )
    print(
        f"Final Dimmer Value: {summary['final_dimmer'] if summary['final_dimmer'] is not None else 'N/A'}"
    )
    final_util = summary["final_utility"]
    print(
        f"Final Utility Value: {final_util:.4f}"
        if final_util is not None
        else "Final Utility Value: N/A"
    )
    avg_util_val = summary["avg_utility"]
    print(
        f"Average Utility: {avg_util_val:.4f}"
        if avg_util_val is not None and not np.isnan(avg_util_val)
        else "Average Utility: N/A"
    )
    print("=" * 50)

    # Save summary to JSON
    summary_file = "/home/prakhar/dev/prakhar479/polaris/swim_experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
