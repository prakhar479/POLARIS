#!/usr/bin/env python3
"""
Simple script to analyze performance metrics from the agentic reasoner.

Now also aggregates token usage (input/output) from LLM API calls
present in the log. Specifically, it looks for `gemini_api_call` entries
and aggregates `input_tokens` and `output_tokens` from their `details`.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import statistics


def analyze_performance_metrics(log_file_path: str):
    """Analyze performance metrics from the log file.

    In addition to timing breakdowns by `metric_type`, this will report token
    usage for LLM calls. It specifically targets `gemini_api_call` entries
    and reports `input_tokens` and `output_tokens`.
    """

    if not Path(log_file_path).exists():
        print(f"Log file not found: {log_file_path}")
        return

    metrics = defaultdict(list)
    # Aggregate token usage across LLM API calls when details are present.
    token_agg = {
        "calls": 0,
        "input_total": 0,
        "output_total": 0,
        # Per-call distributions for stats
        "input_list": [],
        "output_list": [],
        "total_list": [],
        "by_model": defaultdict(
            lambda: {
                "calls": 0,
                "input_total": 0,
                "output_total": 0,
                "input_list": [],
                "output_list": [],
                "total_list": [],
            }
        ),
    }

    # Read and parse metrics
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                metric_type = entry["metric_type"]
                duration_ms = entry["duration_ms"]
                metrics[metric_type].append(duration_ms)

                # Token usage extraction (when present).
                # Modified to specifically look for 'gemini_api_call'
                # based on the provided log structure.
                if metric_type == "gemini_api_call":
                    details = entry.get("details") or {}
                    # Support alternative naming if present.
                    input_tokens = details.get("input_tokens") or details.get("prompt_tokens")
                    output_tokens = details.get("output_tokens") or details.get("completion_tokens")

                    # If token fields exist, treat this as a token-bearing LLM call.
                    if input_tokens is not None or output_tokens is not None:
                        model_name = details.get("model", "unknown")
                        in_tok = int(input_tokens or 0)
                        out_tok = int(output_tokens or 0)
                        tot_tok = in_tok + out_tok  # No reasoning_tokens

                        token_agg["calls"] += 1
                        token_agg["input_total"] += in_tok
                        token_agg["output_total"] += out_tok
                        token_agg["input_list"].append(in_tok)
                        token_agg["output_list"].append(out_tok)
                        token_agg["total_list"].append(tot_tok)

                        model_bucket = token_agg["by_model"][model_name]
                        model_bucket["calls"] += 1
                        model_bucket["input_total"] += in_tok
                        model_bucket["output_total"] += out_tok
                        model_bucket["input_list"].append(in_tok)
                        model_bucket["output_list"].append(out_tok)
                        model_bucket["total_list"].append(tot_tok)
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

    # Token usage summary
    if token_agg["calls"] > 0:
        print()
        print("=== TOKEN USAGE ===")
        total_calls = token_agg["calls"]
        input_total = token_agg["input_total"]
        output_total = token_agg["output_total"]
        grand_total = input_total + output_total

        print(f"LLM Calls With Tokens (gemini_api_call): {total_calls}")
        print(f"  Total input tokens: {input_total}")
        print(f"  Total output tokens: {output_total}")
        print(f"  Grand total tokens: {grand_total}")

        def _stats(vals):
            if not vals:
                return (0, 0, 0, 0)
            return (
                statistics.mean(vals),
                statistics.median(vals),
                min(vals),
                max(vals),
            )

        in_avg, in_med, in_min, in_max = _stats(token_agg["input_list"])
        out_avg, out_med, out_min, out_max = _stats(token_agg["output_list"])
        tot_avg, tot_med, tot_min, tot_max = _stats(token_agg["total_list"])

        print("\nPer-call token stats:")
        print(
            f"  Input tokens -> avg: {in_avg:.2f}, median: {in_med:.2f}, min: {in_min}, max: {in_max}"
        )
        print(
            f"  Output tokens -> avg: {out_avg:.2f}, median: {out_med:.2f}, min: {out_min}, max: {out_max}"
        )
        print(
            f"  Total tokens -> avg: {tot_avg:.2f}, median: {tot_med:.2f}, min: {tot_min}, max: {tot_max}"
        )

        # Optional: per-model breakdown for quick diagnostics.
        if token_agg["by_model"]:
            print()
            print("-- By Model --")
            for model, agg in token_agg["by_model"].items():
                m_calls = agg["calls"]
                m_input = agg["input_total"]
                m_output = agg["output_total"]
                m_total = m_input + m_output
                # compute model-level per-call stats
                m_in_avg, m_in_med, m_in_min, m_in_max = (
                    _stats(agg["input_list"]) if agg["input_list"] else (0, 0, 0, 0)
                )
                m_out_avg, m_out_med, m_out_min, m_out_max = (
                    _stats(agg["output_list"]) if agg["output_list"] else (0, 0, 0, 0)
                )
                m_tot_avg, m_tot_med, m_tot_min, m_tot_max = (
                    _stats(agg["total_list"]) if agg["total_list"] else (0, 0, 0, 0)
                )
                print(f"{model}:")
                print(f"  Calls: {m_calls}")
                print(f"  Input/Output: {m_input}/{m_output}")
                print(f"  Total tokens: {m_total} | Avg/call: {m_tot_avg:.2f}")
                print(
                    f"  Per-call stats -> Input: avg {m_in_avg:.2f}, med {m_in_med:.2f}, min {m_in_min}, max {m_in_max} | "
                    f"Output: avg {m_out_avg:.2f}, med {m_out_med:.2f}, min {m_out_min}, max {m_out_max} | "
                    f"Total: avg {m_tot_avg:.2f}, med {m_tot_med:.2f}, min {m_tot_min}, max {m_tot_max}"
                )


def main():
    """Main function."""

    # Default log file path
    default_path = "logs/overhead1/performance_metrics.jsonl"

    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = default_path

    analyze_performance_metrics(log_file)


if __name__ == "__main__":
    main()
