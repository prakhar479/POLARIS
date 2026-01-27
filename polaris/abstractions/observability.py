"""
Observability interfaces for logging and metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Logger(ABC):
    """Interface for structured logging."""

    @abstractmethod
    def info(self, message: str, **context) -> None:
        """Log info message with context."""
        pass

    @abstractmethod
    def error(self, message: str, **context) -> None:
        """Log error message with context."""
        pass

    @abstractmethod
    def warning(self, message: str, **context) -> None:
        """Log warning message with context."""
        pass

    @abstractmethod
    def debug(self, message: str, **context) -> None:
        """Log debug message with context."""
        pass


class MetricsCollector(ABC):
    """Interface for metrics collection."""

    @abstractmethod
    def increment(
        self,
        metric: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value."""
        pass

    @abstractmethod
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a value in histogram."""
        pass

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        return {}
