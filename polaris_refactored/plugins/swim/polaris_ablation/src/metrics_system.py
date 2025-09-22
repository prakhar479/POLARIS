"""
Comprehensive Metrics Collection and Reporting System

Provides adaptation performance metrics collection, system health monitoring,
component status tracking, and metrics export functionality for analysis.
"""

import asyncio
import time
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import statistics
import csv


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class ComponentStatus(Enum):
    """Component status values."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_name: str
    status: ComponentStatus
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationMetrics:
    """Metrics for adaptation performance."""
    adaptation_id: str
    action_type: str
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    execution_time_ms: float
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collects and stores metrics with time-series support."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def record_metric(self, metric: MetricValue) -> None:
        """Record a metric value."""
        with self.lock:
            metric_key = self._get_metric_key(metric.name, metric.labels)
            self.metrics[metric_key].append(metric)
    
    def record_value(self, 
                    name: str, 
                    value: Union[int, float],
                    metric_type: MetricType = MetricType.GAUGE,
                    labels: Optional[Dict[str, str]] = None,
                    unit: Optional[str] = None,
                    description: Optional[str] = None) -> None:
        """Record a metric value with simplified interface."""
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            unit=unit,
            description=description
        )
        self.record_metric(metric)
    
    def get_metrics(self, 
                   name: str,
                   labels: Optional[Dict[str, str]] = None,
                   since: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[MetricValue]:
        """Get metrics by name and optional filters."""
        with self.lock:
            metric_key = self._get_metric_key(name, labels or {})
            metrics = list(self.metrics.get(metric_key, []))
        
        # Apply time filter
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        # Sort by timestamp
        metrics.sort(key=lambda x: x.timestamp)
        
        # Apply limit
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def get_latest_value(self, 
                        name: str,
                        labels: Optional[Dict[str, str]] = None) -> Optional[MetricValue]:
        """Get the latest value for a metric."""
        metrics = self.get_metrics(name, labels, limit=1)
        return metrics[0] if metrics else None
    
    def get_aggregated_value(self,
                           name: str,
                           aggregation: str = "avg",
                           labels: Optional[Dict[str, str]] = None,
                           since: Optional[datetime] = None) -> Optional[float]:
        """Get aggregated value for a metric."""
        metrics = self.get_metrics(name, labels, since)
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        if aggregation == "avg":
            return statistics.mean(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "median":
            return statistics.median(values)
        elif aggregation == "count":
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_all_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self.lock:
            return list(set(key.split('{')[0] for key in self.metrics.keys()))
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear metrics (all or by name)."""
        with self.lock:
            if name:
                keys_to_remove = [key for key in self.metrics.keys() if key.startswith(name)]
                for key in keys_to_remove:
                    del self.metrics[key]
            else:
                self.metrics.clear()


class AdaptationTracker:
    """Tracks adaptation performance and success rates."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.adaptations: deque = deque(maxlen=max_history)
        self.lock = threading.Lock()
    
    def start_adaptation(self, adaptation_id: str, action_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking an adaptation."""
        adaptation = AdaptationMetrics(
            adaptation_id=adaptation_id,
            action_type=action_type,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            success=False,
            execution_time_ms=0.0,
            context=context or {}
        )
        
        with self.lock:
            self.adaptations.append(adaptation)
    
    def complete_adaptation(self, 
                          adaptation_id: str,
                          success: bool,
                          impact_metrics: Optional[Dict[str, float]] = None,
                          error_message: Optional[str] = None) -> None:
        """Complete tracking an adaptation."""
        with self.lock:
            # Find the adaptation
            for adaptation in reversed(self.adaptations):
                if adaptation.adaptation_id == adaptation_id:
                    adaptation.end_time = datetime.now(timezone.utc)
                    adaptation.success = success
                    adaptation.execution_time_ms = (
                        adaptation.end_time - adaptation.start_time
                    ).total_seconds() * 1000
                    adaptation.impact_metrics = impact_metrics or {}
                    adaptation.error_message = error_message
                    break
    
    def get_adaptations(self, 
                       since: Optional[datetime] = None,
                       action_type: Optional[str] = None,
                       success_only: Optional[bool] = None) -> List[AdaptationMetrics]:
        """Get adaptation history with filters."""
        with self.lock:
            adaptations = list(self.adaptations)
        
        # Apply filters
        if since:
            adaptations = [a for a in adaptations if a.start_time >= since]
        
        if action_type:
            adaptations = [a for a in adaptations if a.action_type == action_type]
        
        if success_only is not None:
            adaptations = [a for a in adaptations if a.success == success_only]
        
        return adaptations
    
    def get_success_rate(self, 
                        since: Optional[datetime] = None,
                        action_type: Optional[str] = None) -> float:
        """Get adaptation success rate."""
        adaptations = self.get_adaptations(since=since, action_type=action_type)
        completed_adaptations = [a for a in adaptations if a.end_time is not None]
        
        if not completed_adaptations:
            return 0.0
        
        successful = sum(1 for a in completed_adaptations if a.success)
        return successful / len(completed_adaptations)
    
    def get_average_execution_time(self, 
                                 since: Optional[datetime] = None,
                                 action_type: Optional[str] = None) -> float:
        """Get average execution time in milliseconds."""
        adaptations = self.get_adaptations(since=since, action_type=action_type)
        completed_adaptations = [a for a in adaptations if a.end_time is not None]
        
        if not completed_adaptations:
            return 0.0
        
        times = [a.execution_time_ms for a in completed_adaptations]
        return statistics.mean(times)
    
    def get_adaptation_statistics(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        adaptations = self.get_adaptations(since=since)
        completed_adaptations = [a for a in adaptations if a.end_time is not None]
        
        if not completed_adaptations:
            return {
                "total_adaptations": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "action_type_distribution": {},
                "error_distribution": {}
            }
        
        # Calculate statistics
        successful = [a for a in completed_adaptations if a.success]
        failed = [a for a in completed_adaptations if not a.success]
        
        action_types = defaultdict(int)
        error_types = defaultdict(int)
        
        for adaptation in completed_adaptations:
            action_types[adaptation.action_type] += 1
            if not adaptation.success and adaptation.error_message:
                error_types[adaptation.error_message] += 1
        
        execution_times = [a.execution_time_ms for a in completed_adaptations]
        
        return {
            "total_adaptations": len(completed_adaptations),
            "successful_adaptations": len(successful),
            "failed_adaptations": len(failed),
            "success_rate": len(successful) / len(completed_adaptations),
            "average_execution_time_ms": statistics.mean(execution_times),
            "min_execution_time_ms": min(execution_times),
            "max_execution_time_ms": max(execution_times),
            "median_execution_time_ms": statistics.median(execution_times),
            "action_type_distribution": dict(action_types),
            "error_distribution": dict(error_types)
        }


class ComponentHealthMonitor:
    """Monitors health status of system components."""
    
    def __init__(self):
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.lock = threading.Lock()
    
    def register_component(self, 
                         component_name: str,
                         health_check: Optional[Callable] = None) -> None:
        """Register a component for health monitoring."""
        with self.lock:
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status=ComponentStatus.UNKNOWN,
                timestamp=datetime.now(timezone.utc)
            )
            
            if health_check:
                self.health_checks[component_name] = health_check
    
    def update_component_health(self,
                              component_name: str,
                              status: ComponentStatus,
                              metrics: Optional[Dict[str, float]] = None,
                              message: Optional[str] = None,
                              details: Optional[Dict[str, Any]] = None) -> None:
        """Update component health status."""
        with self.lock:
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status=status,
                timestamp=datetime.now(timezone.utc),
                metrics=metrics or {},
                message=message,
                details=details or {}
            )
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status of a component."""
        with self.lock:
            return self.component_health.get(component_name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health status of all components."""
        with self.lock:
            return self.component_health.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self.lock:
            components = list(self.component_health.values())
        
        if not components:
            return {"overall_status": "unknown", "component_count": 0}
        
        status_counts = defaultdict(int)
        for component in components:
            status_counts[component.status.value] += 1
        
        # Determine overall status
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        elif status_counts["healthy"] == len(components):
            overall_status = "healthy"
        else:
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "component_count": len(components),
            "status_distribution": dict(status_counts),
            "last_updated": max(c.timestamp for c in components).isoformat()
        }
    
    async def run_health_checks(self) -> None:
        """Run registered health checks for all components."""
        for component_name, health_check in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(health_check):
                    result = await health_check()
                else:
                    result = health_check()
                
                # Parse health check result
                if isinstance(result, dict):
                    status = ComponentStatus(result.get("status", "unknown"))
                    metrics = result.get("metrics", {})
                    message = result.get("message")
                    details = result.get("details", {})
                else:
                    status = ComponentStatus.HEALTHY if result else ComponentStatus.CRITICAL
                    metrics = {}
                    message = None
                    details = {}
                
                self.update_component_health(component_name, status, metrics, message, details)
                
            except Exception as e:
                self.update_component_health(
                    component_name,
                    ComponentStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )


class MetricsExporter:
    """Exports metrics to various formats and destinations."""
    
    def __init__(self, collector: MetricCollector, adaptation_tracker: AdaptationTracker):
        self.collector = collector
        self.adaptation_tracker = adaptation_tracker
    
    def export_to_json(self, output_path: str, since: Optional[datetime] = None) -> None:
        """Export metrics to JSON file."""
        data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "adaptations": []
        }
        
        # Export metrics
        for metric_name in self.collector.get_all_metric_names():
            metrics = self.collector.get_metrics(metric_name, since=since)
            data["metrics"][metric_name] = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value,
                    "type": m.metric_type.value,
                    "labels": m.labels,
                    "unit": m.unit,
                    "description": m.description
                }
                for m in metrics
            ]
        
        # Export adaptations
        adaptations = self.adaptation_tracker.get_adaptations(since=since)
        data["adaptations"] = [
            {
                "adaptation_id": a.adaptation_id,
                "action_type": a.action_type,
                "start_time": a.start_time.isoformat(),
                "end_time": a.end_time.isoformat() if a.end_time else None,
                "success": a.success,
                "execution_time_ms": a.execution_time_ms,
                "impact_metrics": a.impact_metrics,
                "error_message": a.error_message,
                "context": a.context
            }
            for a in adaptations
        ]
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_to_csv(self, output_dir: str, since: Optional[datetime] = None) -> None:
        """Export metrics to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        for metric_name in self.collector.get_all_metric_names():
            metrics = self.collector.get_metrics(metric_name, since=since)
            if not metrics:
                continue
            
            csv_file = output_path / f"{metric_name.replace('/', '_')}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'value', 'type', 'unit', 'labels'])
                
                for metric in metrics:
                    writer.writerow([
                        metric.timestamp.isoformat(),
                        metric.value,
                        metric.metric_type.value,
                        metric.unit or '',
                        json.dumps(metric.labels)
                    ])
        
        # Export adaptations
        adaptations = self.adaptation_tracker.get_adaptations(since=since)
        if adaptations:
            csv_file = output_path / "adaptations.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'adaptation_id', 'action_type', 'start_time', 'end_time',
                    'success', 'execution_time_ms', 'error_message'
                ])
                
                for adaptation in adaptations:
                    writer.writerow([
                        adaptation.adaptation_id,
                        adaptation.action_type,
                        adaptation.start_time.isoformat(),
                        adaptation.end_time.isoformat() if adaptation.end_time else '',
                        adaptation.success,
                        adaptation.execution_time_ms,
                        adaptation.error_message or ''
                    ])
    
    def export_prometheus_format(self, output_path: str) -> None:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name in self.collector.get_all_metric_names():
            latest_metric = self.collector.get_latest_value(metric_name)
            if not latest_metric:
                continue
            
            # Add help and type comments
            if latest_metric.description:
                lines.append(f"# HELP {metric_name} {latest_metric.description}")
            lines.append(f"# TYPE {metric_name} {latest_metric.metric_type.value}")
            
            # Add metric value
            if latest_metric.labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in latest_metric.labels.items())
                lines.append(f"{metric_name}{{{label_str}}} {latest_metric.value}")
            else:
                lines.append(f"{metric_name} {latest_metric.value}")
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))


class SwimPolarisMetricsSystem:
    """
    Comprehensive metrics system for SWIM POLARIS adaptation system.
    
    Provides metrics collection, adaptation tracking, component health monitoring,
    and export capabilities for analysis and observability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics system.
        
        Args:
            config: Metrics system configuration
        """
        self.config = config
        
        # Initialize components
        max_history = config.get('max_history', 10000)
        self.collector = MetricCollector(max_history)
        self.adaptation_tracker = AdaptationTracker(max_history)
        self.health_monitor = ComponentHealthMonitor()
        self.exporter = MetricsExporter(self.collector, self.adaptation_tracker)
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.export_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # System metrics
        self._last_collection_time = time.time()
        self._collection_count = 0
    
    async def start(self) -> None:
        """Start the metrics system."""
        # Start background collection
        if self.config.get('collection_enabled', True):
            collection_interval = self.config.get('collection_interval', 30.0)
            self.collection_task = asyncio.create_task(
                self._collection_loop(collection_interval)
            )
        
        # Start periodic export
        if self.config.get('export_enabled', False):
            export_interval = self.config.get('export_interval', 300.0)
            self.export_task = asyncio.create_task(
                self._export_loop(export_interval)
            )
        
        # Start health checks
        if self.config.get('health_checks_enabled', True):
            health_check_interval = self.config.get('health_check_interval', 60.0)
            self.health_check_task = asyncio.create_task(
                self._health_check_loop(health_check_interval)
            )
    
    async def stop(self) -> None:
        """Stop the metrics system."""
        # Cancel background tasks
        for task in [self.collection_task, self.export_task, self.health_check_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _collection_loop(self, interval: float) -> None:
        """Background metrics collection loop."""
        while True:
            try:
                await self._collect_system_metrics()
                self._collection_count += 1
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Metrics collection error: {e}")
                await asyncio.sleep(interval)
    
    async def _export_loop(self, interval: float) -> None:
        """Background metrics export loop."""
        while True:
            try:
                await self._export_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics export error: {e}")
                await asyncio.sleep(interval)
    
    async def _health_check_loop(self, interval: float) -> None:
        """Background health check loop."""
        while True:
            try:
                await self.health_monitor.run_health_checks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check error: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        current_time = time.time()
        
        # Collection rate
        time_since_last = current_time - self._last_collection_time
        if time_since_last > 0:
            collection_rate = 1.0 / time_since_last
            self.collector.record_value(
                "metrics_collection_rate",
                collection_rate,
                MetricType.GAUGE,
                unit="collections/second"
            )
        
        # Collection count
        self.collector.record_value(
            "metrics_collection_count",
            self._collection_count,
            MetricType.COUNTER,
            unit="collections"
        )
        
        # Memory usage (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.collector.record_value(
                "system_memory_rss",
                memory_info.rss,
                MetricType.GAUGE,
                unit="bytes"
            )
            
            self.collector.record_value(
                "system_memory_vms",
                memory_info.vms,
                MetricType.GAUGE,
                unit="bytes"
            )
            
            self.collector.record_value(
                "system_cpu_percent",
                process.cpu_percent(),
                MetricType.GAUGE,
                unit="percent"
            )
        except ImportError:
            pass  # psutil not available
        
        self._last_collection_time = current_time
    
    async def _export_metrics(self) -> None:
        """Export metrics to configured destinations."""
        export_config = self.config.get('export', {})
        
        if export_config.get('json_enabled', False):
            output_path = export_config.get('json_path', 'metrics/metrics.json')
            self.exporter.export_to_json(output_path)
        
        if export_config.get('csv_enabled', False):
            output_dir = export_config.get('csv_dir', 'metrics/csv')
            self.exporter.export_to_csv(output_dir)
        
        if export_config.get('prometheus_enabled', False):
            output_path = export_config.get('prometheus_path', 'metrics/metrics.prom')
            self.exporter.export_prometheus_format(output_path)
    
    # Public API methods
    
    def record_metric(self, name: str, value: Union[int, float], **kwargs) -> None:
        """Record a metric value."""
        self.collector.record_value(name, value, **kwargs)
    
    def start_adaptation_tracking(self, adaptation_id: str, action_type: str, **context) -> None:
        """Start tracking an adaptation."""
        self.adaptation_tracker.start_adaptation(adaptation_id, action_type, context)
    
    def complete_adaptation_tracking(self, adaptation_id: str, success: bool, **kwargs) -> None:
        """Complete adaptation tracking."""
        self.adaptation_tracker.complete_adaptation(adaptation_id, success, **kwargs)
    
    def register_component(self, component_name: str, health_check: Optional[Callable] = None) -> None:
        """Register a component for health monitoring."""
        self.health_monitor.register_component(component_name, health_check)
    
    def update_component_health(self, component_name: str, status: ComponentStatus, **kwargs) -> None:
        """Update component health status."""
        self.health_monitor.update_component_health(component_name, status, **kwargs)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "system_metrics": {
                "collection_count": self._collection_count,
                "metric_names": self.collector.get_all_metric_names(),
                "total_metrics": len(self.collector.get_all_metric_names())
            },
            "adaptation_metrics": self.adaptation_tracker.get_adaptation_statistics(),
            "health_summary": self.health_monitor.get_system_health_summary()
        }
    
    def export_current_metrics(self, output_path: str, format: str = "json") -> None:
        """Export current metrics to file."""
        if format == "json":
            self.exporter.export_to_json(output_path)
        elif format == "csv":
            self.exporter.export_to_csv(output_path)
        elif format == "prometheus":
            self.exporter.export_prometheus_format(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")