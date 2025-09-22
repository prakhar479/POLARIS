"""
SWIM Metrics Processor

This module processes telemetry data from SWIM, calculates derived metrics,
validates data quality, and prepares metrics for the POLARIS digital twin layer.
"""

import asyncio
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Deque
from enum import Enum

from polaris_refactored.src.domain.models import SystemState, MetricValue, HealthStatus
from polaris_refactored.src.framework.events import TelemetryEvent
from .swim_adaptation_strategies import SwimMetrics


class MetricQuality(Enum):
    """Quality levels for metrics."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class MetricQualityInfo:
    """Information about metric quality."""
    quality: MetricQuality
    confidence: float
    issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProcessedMetrics:
    """Container for processed SWIM metrics."""
    raw_metrics: Dict[str, MetricValue]
    derived_metrics: Dict[str, MetricValue]
    quality_info: Dict[str, MetricQualityInfo]
    swim_metrics: SwimMetrics
    health_status: HealthStatus
    timestamp: datetime
    processing_duration: float


class SwimMetricsValidator:
    """Validates SWIM metrics for quality and consistency."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validation thresholds
        self.response_time_max = config.get("response_time_max", 10000.0)  # 10 seconds
        self.utilization_max = config.get("utilization_max", 1.0)
        self.dimmer_range = config.get("dimmer_range", (0.0, 1.0))
        self.server_count_max = config.get("server_count_max", 100)
    
    def validate_metric(self, name: str, value: Any, 
                       historical_values: Optional[List[Any]] = None) -> MetricQualityInfo:
        """Validate a single metric value."""
        issues = []
        confidence = 1.0
        
        # Basic type and range validation
        if value is None:
            return MetricQualityInfo(MetricQuality.INVALID, 0.0, ["null_value"])
        
        # Metric-specific validation
        if name == "server_count":
            issues.extend(self._validate_server_count(value))
        elif name == "active_servers":
            issues.extend(self._validate_active_servers(value))
        elif name == "max_servers":
            issues.extend(self._validate_max_servers(value))
        elif name == "dimmer":
            issues.extend(self._validate_dimmer(value))
        elif name in ["basic_response_time", "optional_response_time"]:
            issues.extend(self._validate_response_time(value))
        elif name == "server_utilization":
            issues.extend(self._validate_utilization(value))
        
        # Historical consistency validation
        if historical_values:
            issues.extend(self._validate_consistency(name, value, historical_values))
        
        # Determine quality based on issues
        if not issues:
            quality = MetricQuality.EXCELLENT
        elif len(issues) == 1 and "minor" in issues[0]:
            quality = MetricQuality.GOOD
            confidence = 0.8
        elif len(issues) <= 2:
            quality = MetricQuality.ACCEPTABLE
            confidence = 0.6
        else:
            quality = MetricQuality.POOR
            confidence = 0.3
        
        return MetricQualityInfo(quality, confidence, issues)
    
    def _validate_server_count(self, value: Any) -> List[str]:
        """Validate server count metric."""
        issues = []
        
        if not isinstance(value, int):
            issues.append("non_integer_server_count")
        elif value < 0:
            issues.append("negative_server_count")
        elif value > self.server_count_max:
            issues.append("excessive_server_count")
        
        return issues
    
    def _validate_active_servers(self, value: Any) -> List[str]:
        """Validate active servers metric."""
        issues = []
        
        if not isinstance(value, int):
            issues.append("non_integer_active_servers")
        elif value < 0:
            issues.append("negative_active_servers")
        
        return issues
    
    def _validate_max_servers(self, value: Any) -> List[str]:
        """Validate max servers metric."""
        issues = []
        
        if not isinstance(value, int):
            issues.append("non_integer_max_servers")
        elif value <= 0:
            issues.append("invalid_max_servers")
        elif value > self.server_count_max:
            issues.append("excessive_max_servers")
        
        return issues
    
    def _validate_dimmer(self, value: Any) -> List[str]:
        """Validate dimmer metric."""
        issues = []
        
        if not isinstance(value, (int, float)):
            issues.append("non_numeric_dimmer")
        else:
            min_dimmer, max_dimmer = self.dimmer_range
            if value < min_dimmer or value > max_dimmer:
                issues.append("dimmer_out_of_range")
        
        return issues
    
    def _validate_response_time(self, value: Any) -> List[str]:
        """Validate response time metric."""
        issues = []
        
        if not isinstance(value, (int, float)):
            issues.append("non_numeric_response_time")
        elif value < 0:
            issues.append("negative_response_time")
        elif value > self.response_time_max:
            issues.append("excessive_response_time")
        
        return issues
    
    def _validate_utilization(self, value: Any) -> List[str]:
        """Validate utilization metric."""
        issues = []
        
        if not isinstance(value, (int, float)):
            issues.append("non_numeric_utilization")
        elif value < 0 or value > self.utilization_max:
            issues.append("utilization_out_of_range")
        
        return issues
    
    def _validate_consistency(self, name: str, value: Any, 
                            historical_values: List[Any]) -> List[str]:
        """Validate consistency with historical values."""
        issues = []
        
        if not historical_values:
            return issues
        
        try:
            # Check for sudden large changes
            recent_values = historical_values[-5:]  # Last 5 values
            if len(recent_values) >= 2:
                avg_recent = statistics.mean(recent_values)
                
                # Define acceptable change thresholds by metric type
                if name in ["basic_response_time", "optional_response_time"]:
                    threshold = 0.5  # 50% change
                elif name == "server_utilization":
                    threshold = 0.3  # 30% change
                elif name == "dimmer":
                    threshold = 0.2  # 20% change
                else:
                    threshold = 0.4  # 40% change for others
                
                if avg_recent > 0:
                    change_ratio = abs(value - avg_recent) / avg_recent
                    if change_ratio > threshold:
                        issues.append(f"minor_sudden_change_{change_ratio:.2f}")
        
        except (TypeError, ValueError, ZeroDivisionError):
            # If we can't calculate consistency, it's not a critical issue
            pass
        
        return issues


class SwimDerivedMetricsCalculator:
    """Calculates derived metrics from raw SWIM metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_derived_metrics(self, raw_metrics: Dict[str, MetricValue],
                                historical_metrics: List[SwimMetrics]) -> Dict[str, MetricValue]:
        """Calculate all derived metrics."""
        derived = {}
        timestamp = datetime.now(timezone.utc)
        
        # Extract raw values
        server_count = self._get_metric_value(raw_metrics, "server_count")
        active_servers = self._get_metric_value(raw_metrics, "active_servers")
        max_servers = self._get_metric_value(raw_metrics, "max_servers")
        dimmer = self._get_metric_value(raw_metrics, "dimmer")
        basic_rt = self._get_metric_value(raw_metrics, "basic_response_time")
        optional_rt = self._get_metric_value(raw_metrics, "optional_response_time")
        
        # Server utilization
        if active_servers is not None and max_servers is not None and max_servers > 0:
            utilization = active_servers / max_servers
            derived["server_utilization"] = MetricValue(
                name="server_utilization",
                value=utilization,
                unit="ratio",
                timestamp=timestamp,
                tags={"derived": "true", "source": "active_servers,max_servers"}
            )
        
        # Performance ratio (inverse of response time, normalized)
        if basic_rt is not None and basic_rt > 0:
            # Higher is better, normalized to 0-1 range
            performance_ratio = min(1.0, 1000.0 / basic_rt)
            derived["performance_ratio"] = MetricValue(
                name="performance_ratio",
                value=performance_ratio,
                unit="ratio",
                timestamp=timestamp,
                tags={"derived": "true", "source": "basic_response_time"}
            )
        
        # Load factor (combination of utilization and performance)
        if "server_utilization" in derived and "performance_ratio" in derived:
            utilization = derived["server_utilization"].value
            performance = derived["performance_ratio"].value
            # Load factor: high utilization with low performance = high load
            load_factor = utilization * (2.0 - performance)
            derived["load_factor"] = MetricValue(
                name="load_factor",
                value=load_factor,
                unit="ratio",
                timestamp=timestamp,
                tags={"derived": "true", "source": "server_utilization,performance_ratio"}
            )
        
        # QoS efficiency (dimmer vs performance trade-off)
        if dimmer is not None and basic_rt is not None and basic_rt > 0:
            # Efficiency: high dimmer with good response time = efficient
            baseline_rt = 500.0  # Baseline response time
            rt_factor = min(1.0, baseline_rt / basic_rt)
            qos_efficiency = dimmer * rt_factor
            derived["qos_efficiency"] = MetricValue(
                name="qos_efficiency",
                value=qos_efficiency,
                unit="ratio",
                timestamp=timestamp,
                tags={"derived": "true", "source": "dimmer,basic_response_time"}
            )
        
        # Resource efficiency (servers vs performance)
        if server_count is not None and basic_rt is not None and basic_rt > 0:
            # Fewer servers with good performance = more efficient
            baseline_servers = 3
            server_factor = baseline_servers / max(1, server_count)
            rt_factor = min(1.0, 500.0 / basic_rt)
            resource_efficiency = server_factor * rt_factor
            derived["resource_efficiency"] = MetricValue(
                name="resource_efficiency",
                value=min(1.0, resource_efficiency),
                unit="ratio",
                timestamp=timestamp,
                tags={"derived": "true", "source": "server_count,basic_response_time"}
            )
        
        # Trend-based metrics (if historical data available)
        if historical_metrics:
            derived.update(self._calculate_trend_metrics(historical_metrics, timestamp))
        
        return derived
    
    def _get_metric_value(self, metrics: Dict[str, MetricValue], name: str) -> Optional[float]:
        """Safely extract metric value."""
        metric = metrics.get(name)
        if metric and isinstance(metric.value, (int, float)):
            return float(metric.value)
        return None
    
    def _calculate_trend_metrics(self, historical_metrics: List[SwimMetrics], 
                               timestamp: datetime) -> Dict[str, MetricValue]:
        """Calculate trend-based derived metrics."""
        derived = {}
        
        if len(historical_metrics) < 3:
            return derived
        
        # Response time trend
        rt_values = [m.basic_response_time for m in historical_metrics[-10:] 
                    if m.basic_response_time is not None]
        if len(rt_values) >= 3:
            rt_trend = self._calculate_trend(rt_values)
            derived["response_time_trend"] = MetricValue(
                name="response_time_trend",
                value=rt_trend,
                unit="slope",
                timestamp=timestamp,
                tags={"derived": "true", "source": "historical_response_time"}
            )
        
        # Utilization trend
        util_values = [m.server_utilization for m in historical_metrics[-10:] 
                      if m.server_utilization is not None]
        if len(util_values) >= 3:
            util_trend = self._calculate_trend(util_values)
            derived["utilization_trend"] = MetricValue(
                name="utilization_trend",
                value=util_trend,
                unit="slope",
                timestamp=timestamp,
                tags={"derived": "true", "source": "historical_utilization"}
            )
        
        # Stability score (inverse of variance)
        if rt_values:
            rt_variance = statistics.variance(rt_values) if len(rt_values) > 1 else 0
            stability = 1.0 / (1.0 + rt_variance / 1000.0)  # Normalize
            derived["system_stability"] = MetricValue(
                name="system_stability",
                value=stability,
                unit="ratio",
                timestamp=timestamp,
                tags={"derived": "true", "source": "response_time_variance"}
            )
        
        return derived
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) for a series of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


class SwimMetricsProcessor:
    """Main processor for SWIM metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.validator = SwimMetricsValidator(config.get("validation", {}))
        self.calculator = SwimDerivedMetricsCalculator(config.get("derived_metrics", {}))
        
        # Historical data storage
        self.max_history_size = config.get("max_history_size", 1000)
        self.metrics_history: Deque[SwimMetrics] = deque(maxlen=self.max_history_size)
        
        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "validation_failures": 0,
            "processing_errors": 0,
            "avg_processing_time": 0.0
        }
    
    async def process_telemetry_event(self, event: TelemetryEvent) -> ProcessedMetrics:
        """Process a telemetry event from SWIM."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Extract raw metrics
            raw_metrics = event.system_state.metrics
            
            # Validate metrics
            quality_info = {}
            for name, metric in raw_metrics.items():
                historical_values = self._get_historical_values(name)
                quality_info[name] = self.validator.validate_metric(
                    name, metric.value, historical_values
                )
            
            # Calculate derived metrics
            derived_metrics = self.calculator.calculate_derived_metrics(
                raw_metrics, list(self.metrics_history)
            )
            
            # Create SwimMetrics object
            swim_metrics = self._create_swim_metrics(raw_metrics, derived_metrics)
            
            # Determine overall health status
            health_status = self._determine_health_status(quality_info, swim_metrics)
            
            # Update history
            self.metrics_history.append(swim_metrics)
            
            # Calculate processing duration
            end_time = datetime.now(timezone.utc)
            processing_duration = (end_time - start_time).total_seconds()
            
            # Update statistics
            self._update_processing_stats(processing_duration)
            
            # Create processed metrics result
            processed = ProcessedMetrics(
                raw_metrics=raw_metrics,
                derived_metrics=derived_metrics,
                quality_info=quality_info,
                swim_metrics=swim_metrics,
                health_status=health_status,
                timestamp=end_time,
                processing_duration=processing_duration
            )
            
            self.logger.debug(f"Processed metrics in {processing_duration:.3f}s")
            return processed
        
        except Exception as e:
            self.processing_stats["processing_errors"] += 1
            self.logger.error(f"Failed to process telemetry event: {e}")
            raise
    
    def _get_historical_values(self, metric_name: str) -> List[Any]:
        """Get historical values for a specific metric."""
        values = []
        for metrics in list(self.metrics_history)[-10:]:  # Last 10 values
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                if value is not None:
                    values.append(value)
        return values
    
    def _create_swim_metrics(self, raw_metrics: Dict[str, MetricValue],
                           derived_metrics: Dict[str, MetricValue]) -> SwimMetrics:
        """Create SwimMetrics object from raw and derived metrics."""
        
        def get_value(metrics_dict: Dict[str, MetricValue], name: str, default=None):
            metric = metrics_dict.get(name)
            return metric.value if metric else default
        
        return SwimMetrics(
            server_count=int(get_value(raw_metrics, "server_count", 0)),
            active_servers=int(get_value(raw_metrics, "active_servers", 0)),
            max_servers=int(get_value(raw_metrics, "max_servers", 0)),
            dimmer=float(get_value(raw_metrics, "dimmer", 1.0)),
            basic_response_time=get_value(raw_metrics, "basic_response_time"),
            optional_response_time=get_value(raw_metrics, "optional_response_time"),
            server_utilization=get_value(derived_metrics, "server_utilization"),
            timestamp=datetime.now(timezone.utc)
        )
    
    def _determine_health_status(self, quality_info: Dict[str, MetricQualityInfo],
                               swim_metrics: SwimMetrics) -> HealthStatus:
        """Determine overall system health status."""
        
        # Check for invalid metrics
        invalid_count = sum(1 for qi in quality_info.values() 
                          if qi.quality == MetricQuality.INVALID)
        if invalid_count > 0:
            return HealthStatus.UNHEALTHY
        
        # Check for poor quality metrics
        poor_count = sum(1 for qi in quality_info.values() 
                        if qi.quality == MetricQuality.POOR)
        if poor_count > len(quality_info) * 0.3:  # More than 30% poor quality
            return HealthStatus.CRITICAL
        
        # Check system-specific health indicators
        if swim_metrics.server_count == 0:
            return HealthStatus.UNHEALTHY
        
        if (swim_metrics.basic_response_time and 
            swim_metrics.basic_response_time > 2000.0):  # 2 seconds
            return HealthStatus.CRITICAL
        
        if (swim_metrics.server_utilization and 
            swim_metrics.server_utilization > 0.95):
            return HealthStatus.WARNING
        
        # Check overall quality
        avg_confidence = statistics.mean(qi.confidence for qi in quality_info.values())
        if avg_confidence < 0.5:
            return HealthStatus.WARNING
        elif avg_confidence < 0.7:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.HEALTHY
    
    def _update_processing_stats(self, processing_duration: float):
        """Update processing statistics."""
        self.processing_stats["total_processed"] += 1
        
        # Update average processing time
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["avg_processing_time"]
        new_avg = ((current_avg * (total - 1)) + processing_duration) / total
        self.processing_stats["avg_processing_time"] = new_avg
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def get_recent_metrics(self, count: int = 10) -> List[SwimMetrics]:
        """Get recent metrics from history."""
        return list(self.metrics_history)[-count:]
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
        self.logger.info("Metrics history cleared")


class SwimMetricsAggregator:
    """Aggregates SWIM metrics over time windows."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Aggregation windows (in seconds)
        self.windows = config.get("windows", [60, 300, 900])  # 1min, 5min, 15min
        
        # Storage for windowed data
        self.windowed_data: Dict[int, Deque[SwimMetrics]] = {
            window: deque() for window in self.windows
        }
    
    def add_metrics(self, metrics: SwimMetrics):
        """Add metrics to all time windows."""
        current_time = metrics.timestamp
        
        for window_size in self.windows:
            window_data = self.windowed_data[window_size]
            
            # Add new metrics
            window_data.append(metrics)
            
            # Remove old metrics outside the window
            cutoff_time = current_time - timedelta(seconds=window_size)
            while window_data and window_data[0].timestamp < cutoff_time:
                window_data.popleft()
    
    def get_aggregated_metrics(self, window_size: int) -> Optional[Dict[str, Any]]:
        """Get aggregated metrics for a specific time window."""
        if window_size not in self.windowed_data:
            return None
        
        window_data = list(self.windowed_data[window_size])
        if not window_data:
            return None
        
        return {
            "window_size": window_size,
            "sample_count": len(window_data),
            "start_time": window_data[0].timestamp,
            "end_time": window_data[-1].timestamp,
            "aggregations": self._calculate_aggregations(window_data)
        }
    
    def _calculate_aggregations(self, metrics_list: List[SwimMetrics]) -> Dict[str, Any]:
        """Calculate aggregations for a list of metrics."""
        aggregations = {}
        
        # Response time aggregations
        rt_values = [m.basic_response_time for m in metrics_list 
                    if m.basic_response_time is not None]
        if rt_values:
            aggregations["response_time"] = {
                "min": min(rt_values),
                "max": max(rt_values),
                "avg": statistics.mean(rt_values),
                "median": statistics.median(rt_values),
                "std": statistics.stdev(rt_values) if len(rt_values) > 1 else 0
            }
        
        # Utilization aggregations
        util_values = [m.server_utilization for m in metrics_list 
                      if m.server_utilization is not None]
        if util_values:
            aggregations["utilization"] = {
                "min": min(util_values),
                "max": max(util_values),
                "avg": statistics.mean(util_values),
                "median": statistics.median(util_values)
            }
        
        # Server count aggregations
        server_counts = [m.server_count for m in metrics_list]
        if server_counts:
            aggregations["server_count"] = {
                "min": min(server_counts),
                "max": max(server_counts),
                "avg": statistics.mean(server_counts),
                "mode": statistics.mode(server_counts)
            }
        
        # Dimmer aggregations
        dimmer_values = [m.dimmer for m in metrics_list]
        if dimmer_values:
            aggregations["dimmer"] = {
                "min": min(dimmer_values),
                "max": max(dimmer_values),
                "avg": statistics.mean(dimmer_values),
                "median": statistics.median(dimmer_values)
            }
        
        return aggregations