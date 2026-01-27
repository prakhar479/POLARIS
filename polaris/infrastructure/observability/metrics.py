"""
Simple metrics collector implementation.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import defaultdict
from datetime import datetime, timezone

from polaris.abstractions.observability import MetricsCollector as MetricsInterface


class SimpleMetricsCollector(MetricsInterface):
    """
    Simple in-memory metrics collector.

    Stores metrics for querying. For production, integrate with
    Prometheus, DataDog, or similar.
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._last_updated: Dict[str, datetime] = {}

    def increment(
        self,
        metric: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter."""
        key = self._make_key(metric, tags)
        self._counters[key] += value
        self._last_updated[key] = datetime.now(timezone.utc)

    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set gauge value."""
        key = self._make_key(metric, tags)
        self._gauges[key] = value
        self._last_updated[key] = datetime.now(timezone.utc)

    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram value."""
        key = self._make_key(metric, tags)
        self._histograms[key].append(value)

        # Keep only last 1000 values
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

        self._last_updated[key] = datetime.now(timezone.utc)

    def get_summary(self) -> Dict[str, Any]:
        """Get all metrics summary."""
        return {
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {
                k: {
                    'count': len(v),
                    'min': min(v) if v else 0,
                    'max': max(v) if v else 0,
                    'avg': sum(v) / len(v) if v else 0
                }
                for k, v in self._histograms.items()
            }
        }

    def export_to_file(self, file_path: Union[str, Path], format: str = 'json') -> None:
        """
        Export metrics to file.
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            self._export_json(file_path)
        elif format.lower() == 'csv':
            self._export_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, file_path: Path) -> None:
        """Export metrics as JSON."""
        data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': self.get_summary(),
            'last_updated': {k: v.isoformat() for k, v in self._last_updated.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, file_path: Path) -> None:
        """Export metrics as CSV."""
        rows = []
        
        # Add counters
        for metric, value in self._counters.items():
            rows.append({
                'metric_name': metric,
                'metric_type': 'counter',
                'value': value,
                'last_updated': self._last_updated.get(metric, '').isoformat() if self._last_updated.get(metric) else ''
            })
        
        # Add gauges
        for metric, value in self._gauges.items():
            rows.append({
                'metric_name': metric,
                'metric_type': 'gauge',
                'value': value,
                'last_updated': self._last_updated.get(metric, '').isoformat() if self._last_updated.get(metric) else ''
            })
        
        # Add histogram summaries
        for metric, values in self._histograms.items():
            if values:
                rows.append({
                    'metric_name': f"{metric}_count",
                    'metric_type': 'histogram',
                    'value': len(values),
                    'last_updated': self._last_updated.get(metric, '').isoformat() if self._last_updated.get(metric) else ''
                })
                rows.append({
                    'metric_name': f"{metric}_avg",
                    'metric_type': 'histogram',
                    'value': sum(values) / len(values),
                    'last_updated': self._last_updated.get(metric, '').isoformat() if self._last_updated.get(metric) else ''
                })
                rows.append({
                    'metric_name': f"{metric}_min",
                    'metric_type': 'histogram',
                    'value': min(values),
                    'last_updated': self._last_updated.get(metric, '').isoformat() if self._last_updated.get(metric) else ''
                })
                rows.append({
                    'metric_name': f"{metric}_max",
                    'metric_type': 'histogram',
                    'value': max(values),
                    'last_updated': self._last_updated.get(metric, '').isoformat() if self._last_updated.get(metric) else ''
                })
        
        if rows:
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['metric_name', 'metric_type', 'value', 'last_updated'])
                writer.writeheader()
                writer.writerows(rows)

    def _make_key(self, metric: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key with tags."""
        if not tags:
            return metric
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric}{{{tag_str}}}"
