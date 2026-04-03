"""Null-object implementation of MetricsCollector.

Allows all components to call ``self.metrics.increment(...)`` etc. *unconditionally* —
the guards ``if self.metrics:`` are no longer needed.

Usage::

from polaris.infrastructure.observability.null_metrics import NullMetricsCollector

metrics = NullMetricsCollector() metrics.increment("polaris.anything")  # silently
ignored
"""

from typing import Any, Dict, Optional

from polaris.abstractions.observability import MetricsCollector


class NullMetricsCollector(MetricsCollector):
    """A metrics collector that silently discards every call.

    Use this as a drop-in replacement for ``None`` so that components do not need ``if
    self.metrics:`` guards before every metric emission.
    """

    def increment(
        self, metric: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """No-op."""

    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """No-op."""

    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """No-op."""

    def get_summary(self) -> Dict[str, Any]:
        """Return an empty summary dict."""
        return {}
