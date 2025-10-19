#!/usr/bin/env python3
"""
Simple script to analyze performance metrics from the agentic reasoner.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import statistics


def analyze_performance_metrics(log_file_path: str):
    """Analyze performance metrics from the log file."""

    if not Path(log_file_path).exists():
        print(f"Log file not found: {log_file_path}")
        return

    metrics = defaultdict(list)

    # Read and parse metrics
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                metric_type = entry["metric_type"]
                duration_ms = entry["duration_ms"]
                metrics[metric_type].append(duration_ms)
            except json.JSONDecodeError:
                continue

    if not metrics:
        print("No metrics found in log file")
        return

    print("=== POLARIS Agentic Reasoner Performance Analysis ===\n")

    # Overall summary
    for metric_type, durations in metrics.items():
        count = len(durations)
        avg_ms = statistics.mean(durations)
        min_ms = min(durations)
        max_ms = max(durations)
        median_ms = statistics.median(durations)

        print(f"{metric_type.upper().replace('_', ' ')}:")
        print(f"  Count: {count}")
        print(f"  Average: {avg_ms:.2f}ms")
        print(f"  Median: {median_ms:.2f}ms")
        print(f"  Min: {min_ms:.2f}ms")
        print(f"  Max: {max_ms:.2f}ms")
        print()

    # Key insights
    print("=== KEY INSIGHTS ===")

    if "end_to_end_reasoning" in metrics:
        end_to_end_times = metrics["end_to_end_reasoning"]
        print(f"End-to-End Response Time: {statistics.mean(end_to_end_times):.2f}ms avg")

    if "gemini_api_call" in metrics:
        gemini_times = metrics["gemini_api_call"]
        print(f"LLM (Gemini) Response Time: {statistics.mean(gemini_times):.2f}ms avg")

    if "kb_query" in metrics:
        kb_times = metrics["kb_query"]
        print(f"Knowledge Base Query Time: {statistics.mean(kb_times):.2f}ms avg")

    if "dt_operation" in metrics:
        dt_times = metrics["dt_operation"]
        print(f"Digital Twin Operation Time: {statistics.mean(dt_times):.2f}ms avg")

    if "tool_call_total" in metrics:
        tool_times = metrics["tool_call_total"]
        print(f"Tool Call Overhead: {statistics.mean(tool_times):.2f}ms avg")


def main():
    """Main function."""

    # Default log file path
    default_path = "logs/overhead/performance_metrics.jsonl"

    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = default_path

    analyze_performance_metrics(log_file)


if __name__ == "__main__":
    main()
