"""Utility functions for POLARIS tools.

This module provides common utility functions for tools, such as argument validation,
time window calculation, and metric extraction.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from polaris.core.models import MetricValue, SystemState


class ToolError(Exception):
    """Structured error information for tool execution failures.

    Note: This is imported here mostly for typing, but real ToolError is defined in
    base.py.
    """


def clamp_int(value: Any, min_val: int, max_val: int, default: int) -> int:
    """Clamp integer arguments to valid range.

    Args:
        value: The value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default if value is None or invalid

    Returns:
        Clamped integer value
    """
    try:
        val = int(value) if value is not None else default
        return max(min_val, min(val, max_val))
    except (TypeError, ValueError):
        return default


def get_time_window(
    args: Dict[str, Any],
    default_seconds: int = 600,
    max_seconds: int = 3600,
) -> Tuple[datetime, datetime]:
    """Calculate time window from arguments.

    Args:
        args: Tool arguments containing window_seconds
        default_seconds: Default window if not specified
        max_seconds: Maximum allowed window

    Returns:
        Tuple of (start_time, end_time) as UTC datetimes
    """
    window_seconds = clamp_int(args.get("window_seconds"), 1, max_seconds, default_seconds)
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=window_seconds)
    return start, end


def extract_metric_values(states: List[SystemState], metric_name: str) -> List[float]:
    """Extract numeric metric values from states.

    Args:
        states: List of system states
        metric_name: Name of metric to extract

    Returns:
        List of float values for the metric
    """
    values: List[float] = []
    for s in states:
        metric_value: Optional[MetricValue] = s.metrics.get(metric_name)
        if metric_value is None or metric_value.value is None:
            continue
        try:
            values.append(float(metric_value.value))
        except (TypeError, ValueError):
            continue
    return values
