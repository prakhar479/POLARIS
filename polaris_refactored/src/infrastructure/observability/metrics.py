"""
Metrics Collection System for POLARIS

Provides comprehensive metrics collection with counters, gauges, and histograms.
Includes business and technical metrics with Prometheus export capabilities.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta


class MetricType(Enum):
    """Types of metrics supported by POLARIS"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Represents a metric value with metadata"""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class HistogramBucket:
    """Histogram bucket for tracking value distributions"""
    upper_bound: float
    count: int = 0


class Metric(ABC):
    """Abstract base class for all metrics"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.created_at = datetime.utcnow()
        self._lock = Lock()
    
    @abstractmethod
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """Get the current metric value"""
        pass
    
    @abstractmethod
    def get_type(self) -> MetricType:
        """Get the metric type"""
        pass


class Counter(Metric):
    """Counter metric that only increases"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._values: Dict[str, float] = defaultdict(float)
    
    def _get_key(self, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate key for label combination"""
        if not labels:
            return ""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter"""
        if amount < 0:
            raise ValueError("Counter can only be incremented with positive values")
        
        key = self._get_key(labels)
        with self._lock:
            self._values[key] += amount
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """Get current counter value"""
        key = self._get_key(labels)
        with self._lock:
            value = self._values[key]
        return MetricValue(value=value, timestamp=datetime.utcnow(), labels=labels or {})
    
    def get_type(self) -> MetricType:
        return MetricType.COUNTER


class Gauge(Metric):
    """Gauge metric that can increase or decrease"""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self._values: Dict[str, float] = defaultdict(float)
    
    def _get_key(self, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate key for label combination"""
        if not labels:
            return ""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge value"""
        key = self._get_key(labels)
        with self._lock:
            self._values[key] = value
    
    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge"""
        key = self._get_key(labels)
        with self._lock:
            self._values[key] += amount
    
    def decrement(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge"""
        key = self._get_key(labels)
        with self._lock:
            self._values[key] -= amount
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """Get current gauge value"""
        key = self._get_key(labels)
        with self._lock:
            value = self._values[key]
        return MetricValue(value=value, timestamp=datetime.utcnow(), labels=labels or {})
    
    def get_type(self) -> MetricType:
        return MetricType.GAUGE


class Histogram(Metric):
    """Histogram metric for tracking value distributions"""
    
    def __init__(self, name: str, description: str, buckets: Optional[List[float]] = None, labels: Optional[List[str]] = None):
        super().__init__(name, description, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self._bucket_counts: Dict[str, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
    
    def _get_key(self, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate key for label combination"""
        if not labels:
            return ""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in the histogram"""
        key = self._get_key(labels)
        
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1
    
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> MetricValue:
        """Get histogram statistics"""
        key = self._get_key(labels)
        
        with self._lock:
            count = self._counts[key]
            sum_value = self._sums[key]
            buckets = dict(self._bucket_counts[key])
        
        value = {
            'count': count,
            'sum': sum_value,
            'buckets': buckets
        }
        
        return MetricValue(value=value, timestamp=datetime.utcnow(), labels=labels or {})
    
    def get_type(self) -> MetricType:
        return MetricType.HISTOGRAM


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        self.histogram = histogram
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration, self.labels)


class PolarisMetricsCollector:
    """
    Central metrics collector for POLARIS system.
    
    Provides business and technical metrics for monitoring system health,
    performance, and behavior. Supports multiple export formats including Prometheus.
    """
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = Lock()
        
        # Initialize core POLARIS metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core POLARIS metrics"""
        
        # System Health Metrics
        self.register_gauge(
            "polaris_system_health_score",
            "Overall health score of managed systems (0-1)",
            ["system_id"]
        )
        
        self.register_gauge(
            "polaris_active_systems_count", 
            "Number of currently active managed systems"
        )
        
        # Adaptation Metrics
        self.register_counter(
            "polaris_adaptations_triggered_total",
            "Total number of adaptations triggered",
            ["system_id", "adaptation_type", "trigger_reason"]
        )
        
        self.register_counter(
            "polaris_adaptations_successful_total",
            "Total number of successful adaptations",
            ["system_id", "adaptation_type"]
        )
        
        self.register_counter(
            "polaris_adaptations_failed_total", 
            "Total number of failed adaptations",
            ["system_id", "adaptation_type", "failure_reason"]
        )
        
        self.register_histogram(
            "polaris_adaptation_duration_seconds",
            "Time taken to complete adaptations",
            labels=["system_id", "adaptation_type"]
        )
        
        # Telemetry Metrics
        self.register_counter(
            "polaris_telemetry_events_received_total",
            "Total telemetry events received",
            ["system_id", "event_type"]
        )
        
        self.register_histogram(
            "polaris_telemetry_processing_duration_seconds",
            "Time taken to process telemetry events",
            labels=["system_id"]
        )
        
        self.register_gauge(
            "polaris_telemetry_queue_size",
            "Current size of telemetry processing queue",
            ["system_id"]
        )
        
        # World Model Metrics
        self.register_counter(
            "polaris_world_model_updates_total",
            "Total world model updates",
            ["system_id", "model_type"]
        )
        
        self.register_histogram(
            "polaris_world_model_prediction_accuracy",
            "Accuracy of world model predictions (0-1)",
            labels=["system_id", "model_type"]
        )
        
        # Learning Metrics
        self.register_counter(
            "polaris_learning_episodes_total",
            "Total learning episodes completed",
            ["system_id", "learning_strategy"]
        )
        
        self.register_gauge(
            "polaris_learning_performance_score",
            "Current learning performance score",
            ["system_id", "learning_strategy"]
        )
        
        # Infrastructure Metrics
        self.register_histogram(
            "polaris_message_bus_latency_seconds",
            "Message bus operation latency",
            labels=["operation", "message_type"]
        )
        
        self.register_counter(
            "polaris_circuit_breaker_state_changes_total",
            "Circuit breaker state changes",
            ["service", "from_state", "to_state"]
        )
        
        self.register_counter(
            "polaris_retry_attempts_total",
            "Total retry attempts",
            ["operation", "success"]
        )
    
    def register_counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Counter:
        """Register a new counter metric"""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            counter = Counter(name, description, labels)
            self._metrics[name] = counter
            return counter
    
    def register_gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Gauge:
        """Register a new gauge metric"""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            gauge = Gauge(name, description, labels)
            self._metrics[name] = gauge
            return gauge
    
    def register_histogram(self, name: str, description: str, buckets: Optional[List[float]] = None, labels: Optional[List[str]] = None) -> Histogram:
        """Register a new histogram metric"""
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name} already exists")
            
            histogram = Histogram(name, description, buckets, labels)
            self._metrics[name] = histogram
            return histogram
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name"""
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics"""
        return dict(self._metrics)
    
    # Convenience methods for core metrics
    def increment_adaptations_triggered(self, system_id: str, adaptation_type: str, trigger_reason: str) -> None:
        """Increment adaptations triggered counter"""
        counter = self.get_metric("polaris_adaptations_triggered_total")
        if isinstance(counter, Counter):
            counter.increment(labels={
                "system_id": system_id,
                "adaptation_type": adaptation_type, 
                "trigger_reason": trigger_reason
            })
    
    def increment_adaptations_successful(self, system_id: str, adaptation_type: str) -> None:
        """Increment successful adaptations counter"""
        counter = self.get_metric("polaris_adaptations_successful_total")
        if isinstance(counter, Counter):
            counter.increment(labels={
                "system_id": system_id,
                "adaptation_type": adaptation_type
            })
    
    def increment_adaptations_failed(self, system_id: str, adaptation_type: str, failure_reason: str) -> None:
        """Increment failed adaptations counter"""
        counter = self.get_metric("polaris_adaptations_failed_total")
        if isinstance(counter, Counter):
            counter.increment(labels={
                "system_id": system_id,
                "adaptation_type": adaptation_type,
                "failure_reason": failure_reason
            })
    
    def set_system_health_score(self, system_id: str, score: float) -> None:
        """Set system health score"""
        gauge = self.get_metric("polaris_system_health_score")
        if isinstance(gauge, Gauge):
            gauge.set(score, labels={"system_id": system_id})
    
    def set_active_systems_count(self, count: int) -> None:
        """Set active systems count"""
        gauge = self.get_metric("polaris_active_systems_count")
        if isinstance(gauge, Gauge):
            gauge.set(count)
    
    def time_adaptation(self, system_id: str, adaptation_type: str) -> Timer:
        """Get timer for adaptation duration"""
        histogram = self.get_metric("polaris_adaptation_duration_seconds")
        if isinstance(histogram, Histogram):
            return Timer(histogram, labels={
                "system_id": system_id,
                "adaptation_type": adaptation_type
            })
        return Timer(Histogram("dummy", "dummy"))
    
    def time_telemetry_processing(self, system_id: str) -> Timer:
        """Get timer for telemetry processing"""
        histogram = self.get_metric("polaris_telemetry_processing_duration_seconds")
        if isinstance(histogram, Histogram):
            return Timer(histogram, labels={"system_id": system_id})
        return Timer(Histogram("dummy", "dummy"))

    # Compatibility helper for code that expects a generic increment_counter API
    def increment_counter(self, name: str, labels: Optional[dict] = None) -> None:
        """Increment a counter metric by name. Safe no-op if metric not registered.

        This method is provided for compatibility: some components expect a generic
        increment_counter(name, labels) helper on the global metrics collector.
        """
        metric = self.get_metric(name)
        if isinstance(metric, Counter):
            metric.increment(labels=labels or {})
        else:
            # Metric not found or not a counter: log a debug and continue (non-fatal)
            pass


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters"""
    
    @abstractmethod
    def export(self, metrics: Dict[str, Metric]) -> str:
        """Export metrics in the target format"""
        pass


class PrometheusExporter(MetricsExporter):
    """Prometheus format metrics exporter"""
    
    def export(self, metrics: Dict[str, Metric]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        for name, metric in metrics.items():
            # Add help and type comments
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.get_type().value}")
            
            if isinstance(metric, (Counter, Gauge)):
                value = metric.get_value()
                labels_str = self._format_labels(value.labels)
                lines.append(f"{name}{labels_str} {value.value}")
                
            elif isinstance(metric, Histogram):
                # Export histogram buckets, count, and sum
                value = metric.get_value()
                labels = value.labels
                hist_data = value.value
                
                # Buckets
                for bucket, count in hist_data.get('buckets', {}).items():
                    bucket_labels = {**labels, 'le': str(bucket)}
                    labels_str = self._format_labels(bucket_labels)
                    lines.append(f"{name}_bucket{labels_str} {count}")
                
                # Count and sum
                labels_str = self._format_labels(labels)
                lines.append(f"{name}_count{labels_str} {hist_data.get('count', 0)}")
                lines.append(f"{name}_sum{labels_str} {hist_data.get('sum', 0)}")
        
        return '\n'.join(lines)
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output"""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"


# Global metrics collector instance
_metrics_collector: Optional[PolarisMetricsCollector] = None


def get_metrics_collector() -> PolarisMetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = PolarisMetricsCollector()
    return _metrics_collector