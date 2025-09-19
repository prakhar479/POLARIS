# Monitoring & Telemetry

## Overview

POLARIS provides comprehensive monitoring and telemetry capabilities that enable real-time system observation, performance tracking, and intelligent decision-making. The monitoring system is designed to be flexible, scalable, and capable of handling diverse system types and metrics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring & Telemetry                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Data Collection │ │ Data Processing │ │ Data Storage    │   │
│  │ - Pull Strategy │ │ - Validation    │ │ - Time Series   │   │
│  │ - Push Strategy │ │ - Transformation│ │ - Aggregation   │   │
│  │ - Batch Strategy│ │ - Enrichment    │ │ - Retention     │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Health Monitoring│ │ Alerting System │ │ Visualization   │   │
│  │ - Status Checks │ │ - Thresholds    │ │ - Dashboards    │   │
│  │ - Anomaly Detect│ │ - Notifications │ │ - Real-time     │   │
│  │ - Trend Analysis│ │ - Escalation    │ │ - Historical    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Collection Strategies

### Pull-Based Collection

**Purpose**: Actively polls managed systems for metrics at regular intervals.

**Implementation**:
```python
class PullCollectionStrategy(MetricCollectionStrategy):
    """Pull-based metric collection strategy."""
    
    def __init__(self, config: PullCollectionConfig):
        self._config = config
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start_collection(self, target: MonitoringTarget) -> None:
        """Start collecting metrics from target."""
        if target.system_id in self._collection_tasks:
            return
        
        task = asyncio.create_task(self._collection_loop(target))
        self._collection_tasks[target.system_id] = task
        
        logger.info(f"Started pull collection for {target.system_id}")
    
    async def stop_collection(self, target: MonitoringTarget) -> None:
        """Stop collecting metrics from target."""
        if target.system_id in self._collection_tasks:
            task = self._collection_tasks[target.system_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._collection_tasks[target.system_id]
        
        logger.info(f"Stopped pull collection for {target.system_id}")
    
    async def _collection_loop(self, target: MonitoringTarget) -> None:
        """Main collection loop for a target."""
        connector = target.connector
        interval = target.collection_interval or self._config.default_interval
        
        while self._running:
            try:
                # Collect metrics
                start_time = time.time()
                metrics = await connector.collect_metrics()
                collection_time = time.time() - start_time
                
                # Create system state
                system_state = SystemState(
                    system_id=target.system_id,
                    timestamp=datetime.utcnow(),
                    metrics=metrics,
                    health_status=await connector.get_health_status()
                )
                
                # Publish telemetry event
                telemetry_event = TelemetryEvent(system_state)
                await self._event_bus.publish(telemetry_event)
                
                # Record collection metrics
                self._record_collection_metrics(target.system_id, collection_time, True)
                
                # Wait for next collection
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Collection failed for {target.system_id}: {e}")
                self._record_collection_metrics(target.system_id, 0, False)
                
                # Exponential backoff on errors
                await asyncio.sleep(min(interval * 2, 300))
    
    def _record_collection_metrics(self, system_id: str, 
                                  collection_time: float, success: bool) -> None:
        """Record collection performance metrics."""
        status = "success" if success else "error"
        
        collection_duration.labels(
            system_id=system_id,
            strategy="pull"
        ).observe(collection_time)
        
        collection_count.labels(
            system_id=system_id,
            strategy="pull",
            status=status
        ).inc()
```

**Configuration**:
```yaml
monitoring:
  pull_collection:
    default_interval: 30  # seconds
    timeout: 10  # seconds
    retry_attempts: 3
    retry_delay: 5  # seconds
    batch_size: 10  # systems to collect concurrently
```

### Push-Based Collection

**Purpose**: Receives metrics pushed from managed systems in real-time.

**Implementation**:
```python
class PushCollectionStrategy(MetricCollectionStrategy):
    """Push-based metric collection strategy."""
    
    def __init__(self, config: PushCollectionConfig):
        self._config = config
        self._endpoints: Dict[str, PushEndpoint] = {}
        self._running = False
    
    async def setup_push_endpoint(self, target: MonitoringTarget) -> None:
        """Set up push endpoint for target system."""
        endpoint = PushEndpoint(
            system_id=target.system_id,
            port=self._config.base_port + hash(target.system_id) % 1000,
            authentication=self._config.authentication
        )
        
        await endpoint.start()
        self._endpoints[target.system_id] = endpoint
        
        # Register message handler
        endpoint.on_message(self._handle_pushed_metrics)
        
        logger.info(f"Push endpoint ready for {target.system_id} on port {endpoint.port}")
    
    async def _handle_pushed_metrics(self, system_id: str, 
                                   metrics_data: Dict[str, Any]) -> None:
        """Handle metrics pushed from system."""
        try:
            # Parse and validate metrics
            metrics = self._parse_metrics(metrics_data)
            
            # Determine health status
            health_status = self._assess_health_status(metrics)
            
            # Create system state
            system_state = SystemState(
                system_id=system_id,
                timestamp=datetime.utcnow(),
                metrics=metrics,
                health_status=health_status
            )
            
            # Publish telemetry event
            telemetry_event = TelemetryEvent(system_state)
            await self._event_bus.publish(telemetry_event)
            
            # Record metrics
            push_received_count.labels(system_id=system_id).inc()
            
        except Exception as e:
            logger.error(f"Failed to process pushed metrics from {system_id}: {e}")
            push_error_count.labels(system_id=system_id).inc()
    
    def _parse_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, MetricValue]:
        """Parse raw metrics data into MetricValue objects."""
        metrics = {}
        
        for name, data in metrics_data.items():
            if isinstance(data, dict):
                # Structured metric with metadata
                metrics[name] = MetricValue(
                    name=name,
                    value=data["value"],
                    unit=data.get("unit", ""),
                    timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
                    tags=data.get("tags", {})
                )
            else:
                # Simple numeric value
                metrics[name] = MetricValue(
                    name=name,
                    value=float(data),
                    unit="",
                    timestamp=datetime.utcnow()
                )
        
        return metrics
```

### Batch Collection

**Purpose**: Collects metrics from multiple systems in batches for efficiency.

**Implementation**:
```python
class BatchCollectionStrategy(MetricCollectionStrategy):
    """Batch-based metric collection strategy."""
    
    def __init__(self, config: BatchCollectionConfig):
        self._config = config
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_batch_processing(self) -> None:
        """Start batch processing task."""
        self._running = True
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
    
    async def collect_batch(self, targets: List[MonitoringTarget]) -> List[TelemetryEvent]:
        """Collect metrics from multiple targets in batch."""
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting batch collection {batch_id} for {len(targets)} targets")
        
        # Create collection tasks
        tasks = []
        for target in targets:
            task = asyncio.create_task(self._collect_single_target(target))
            tasks.append(task)
        
        # Wait for all collections with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.batch_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Batch collection {batch_id} timed out")
            results = [None] * len(targets)
        
        # Process results
        telemetry_events = []
        successful_collections = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Collection failed for {targets[i].system_id}: {result}")
            elif result is not None:
                telemetry_events.append(result)
                successful_collections += 1
        
        # Record batch metrics
        batch_duration = time.time() - start_time
        batch_collection_duration.labels(
            batch_size=len(targets)
        ).observe(batch_duration)
        
        batch_success_rate.labels(
            batch_size=len(targets)
        ).observe(successful_collections / len(targets))
        
        logger.info(
            f"Batch collection {batch_id} completed: "
            f"{successful_collections}/{len(targets)} successful in {batch_duration:.2f}s"
        )
        
        return telemetry_events
    
    async def _collect_single_target(self, target: MonitoringTarget) -> Optional[TelemetryEvent]:
        """Collect metrics from a single target."""
        try:
            metrics = await target.connector.collect_metrics()
            health_status = await target.connector.get_health_status()
            
            system_state = SystemState(
                system_id=target.system_id,
                timestamp=datetime.utcnow(),
                metrics=metrics,
                health_status=health_status
            )
            
            return TelemetryEvent(system_state)
            
        except Exception as e:
            logger.error(f"Failed to collect from {target.system_id}: {e}")
            return None
```

## Data Processing Pipeline

### Metric Validation

**Purpose**: Ensures collected metrics meet quality standards.

```python
class MetricValidator:
    """Validates metric data quality and consistency."""
    
    def __init__(self, config: ValidationConfig):
        self._config = config
        self._validation_rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def validate_metrics(self, metrics: Dict[str, MetricValue]) -> ValidationResult:
        """Validate a collection of metrics."""
        errors = []
        warnings = []
        
        for rule in self._validation_rules:
            result = rule.validate(metrics)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _setup_default_rules(self) -> None:
        """Set up default validation rules."""
        self._validation_rules = [
            RangeValidationRule("cpu_usage", 0.0, 1.0),
            RangeValidationRule("memory_usage", 0.0, 1.0),
            RangeValidationRule("disk_usage", 0.0, 1.0),
            NonNegativeValidationRule(["request_count", "error_count"]),
            TimestampValidationRule(max_age_minutes=5),
            RequiredMetricsRule(["cpu_usage", "memory_usage"])
        ]

class RangeValidationRule(ValidationRule):
    """Validates metric values are within expected range."""
    
    def __init__(self, metric_name: str, min_value: float, max_value: float):
        self.metric_name = metric_name
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, metrics: Dict[str, MetricValue]) -> ValidationResult:
        """Validate metric is within range."""
        errors = []
        
        if self.metric_name in metrics:
            value = metrics[self.metric_name].value
            if not self.min_value <= value <= self.max_value:
                errors.append(
                    f"Metric {self.metric_name} value {value} is outside "
                    f"valid range [{self.min_value}, {self.max_value}]"
                )
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

### Data Transformation

**Purpose**: Transforms and enriches raw metrics data.

```python
class MetricTransformer:
    """Transforms and enriches metric data."""
    
    def __init__(self, config: TransformationConfig):
        self._config = config
        self._transformations: List[MetricTransformation] = []
        self._setup_transformations()
    
    def transform_metrics(self, metrics: Dict[str, MetricValue]) -> Dict[str, MetricValue]:
        """Apply transformations to metrics."""
        transformed_metrics = metrics.copy()
        
        for transformation in self._transformations:
            transformed_metrics = transformation.apply(transformed_metrics)
        
        return transformed_metrics
    
    def _setup_transformations(self) -> None:
        """Set up metric transformations."""
        self._transformations = [
            UnitConversionTransformation(),
            DerivedMetricTransformation(),
            NormalizationTransformation(),
            EnrichmentTransformation()
        ]

class DerivedMetricTransformation(MetricTransformation):
    """Creates derived metrics from existing ones."""
    
    def apply(self, metrics: Dict[str, MetricValue]) -> Dict[str, MetricValue]:
        """Apply derived metric calculations."""
        result = metrics.copy()
        
        # Calculate CPU utilization percentage
        if "cpu_usage" in metrics:
            cpu_usage = metrics["cpu_usage"]
            result["cpu_utilization_percent"] = MetricValue(
                name="cpu_utilization_percent",
                value=cpu_usage.value * 100,
                unit="percent",
                timestamp=cpu_usage.timestamp,
                tags=cpu_usage.tags
            )
        
        # Calculate memory pressure
        if "memory_usage" in metrics and "memory_available" in metrics:
            memory_usage = metrics["memory_usage"].value
            memory_available = metrics["memory_available"].value
            
            if memory_available > 0:
                pressure = memory_usage / (memory_usage + memory_available)
                result["memory_pressure"] = MetricValue(
                    name="memory_pressure",
                    value=pressure,
                    unit="ratio",
                    timestamp=metrics["memory_usage"].timestamp
                )
        
        # Calculate error rate
        if "request_count" in metrics and "error_count" in metrics:
            requests = metrics["request_count"].value
            errors = metrics["error_count"].value
            
            if requests > 0:
                error_rate = errors / requests
                result["error_rate"] = MetricValue(
                    name="error_rate",
                    value=error_rate,
                    unit="ratio",
                    timestamp=metrics["request_count"].timestamp
                )
        
        return result
```

## Health Monitoring

### Health Assessment

**Purpose**: Determines system health based on metrics and thresholds.

```python
class HealthAssessor:
    """Assesses system health based on metrics and rules."""
    
    def __init__(self, config: HealthAssessmentConfig):
        self._config = config
        self._health_rules: List[HealthRule] = []
        self._setup_health_rules()
    
    def assess_health(self, system_state: SystemState) -> HealthAssessment:
        """Assess overall system health."""
        assessments = []
        
        for rule in self._health_rules:
            assessment = rule.assess(system_state)
            assessments.append(assessment)
        
        # Determine overall health (worst case)
        overall_status = HealthStatus.HEALTHY
        issues = []
        
        for assessment in assessments:
            if assessment.status.severity_level() > overall_status.severity_level():
                overall_status = assessment.status
            
            if assessment.issues:
                issues.extend(assessment.issues)
        
        return HealthAssessment(
            system_id=system_state.system_id,
            status=overall_status,
            issues=issues,
            timestamp=datetime.utcnow(),
            rule_assessments=assessments
        )
    
    def _setup_health_rules(self) -> None:
        """Set up health assessment rules."""
        self._health_rules = [
            ThresholdHealthRule("cpu_usage", warning_threshold=0.8, critical_threshold=0.95),
            ThresholdHealthRule("memory_usage", warning_threshold=0.85, critical_threshold=0.95),
            ThresholdHealthRule("disk_usage", warning_threshold=0.9, critical_threshold=0.98),
            ErrorRateHealthRule(warning_threshold=0.01, critical_threshold=0.05),
            ResponseTimeHealthRule(warning_threshold=1000, critical_threshold=5000),
            AvailabilityHealthRule(warning_threshold=0.99, critical_threshold=0.95)
        ]

class ThresholdHealthRule(HealthRule):
    """Health rule based on metric thresholds."""
    
    def __init__(self, metric_name: str, warning_threshold: float, critical_threshold: float):
        self.metric_name = metric_name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def assess(self, system_state: SystemState) -> HealthRuleAssessment:
        """Assess health based on metric threshold."""
        if not system_state.has_metric(self.metric_name):
            return HealthRuleAssessment(
                rule_name=f"threshold_{self.metric_name}",
                status=HealthStatus.UNKNOWN,
                message=f"Metric {self.metric_name} not available"
            )
        
        value = system_state.get_metric_value(self.metric_name)
        
        if value >= self.critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"{self.metric_name} is critical: {value:.2f} >= {self.critical_threshold}"
        elif value >= self.warning_threshold:
            status = HealthStatus.WARNING
            message = f"{self.metric_name} is elevated: {value:.2f} >= {self.warning_threshold}"
        else:
            status = HealthStatus.HEALTHY
            message = f"{self.metric_name} is normal: {value:.2f}"
        
        return HealthRuleAssessment(
            rule_name=f"threshold_{self.metric_name}",
            status=status,
            message=message,
            metric_value=value
        )
```

### Anomaly Detection

**Purpose**: Detects unusual patterns in system metrics.

```python
class AnomalyDetector:
    """Detects anomalies in system metrics using statistical methods."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self._config = config
        self._detectors: Dict[str, MetricAnomalyDetector] = {}
        self._baseline_data: Dict[str, List[float]] = {}
    
    def detect_anomalies(self, system_state: SystemState) -> List[Anomaly]:
        """Detect anomalies in system state."""
        anomalies = []
        
        for metric_name, metric in system_state.metrics.items():
            detector = self._get_detector(metric_name)
            anomaly = detector.detect(metric.value)
            
            if anomaly:
                anomalies.append(Anomaly(
                    system_id=system_state.system_id,
                    metric_name=metric_name,
                    value=metric.value,
                    anomaly_type=anomaly.type,
                    severity=anomaly.severity,
                    confidence=anomaly.confidence,
                    timestamp=system_state.timestamp
                ))
        
        return anomalies
    
    def _get_detector(self, metric_name: str) -> MetricAnomalyDetector:
        """Get or create anomaly detector for metric."""
        if metric_name not in self._detectors:
            self._detectors[metric_name] = StatisticalAnomalyDetector(
                metric_name=metric_name,
                config=self._config
            )
        
        return self._detectors[metric_name]

class StatisticalAnomalyDetector(MetricAnomalyDetector):
    """Statistical anomaly detector using z-score and moving averages."""
    
    def __init__(self, metric_name: str, config: AnomalyDetectionConfig):
        self.metric_name = metric_name
        self._config = config
        self._history: deque = deque(maxlen=config.history_size)
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
    
    def detect(self, value: float) -> Optional[AnomalyResult]:
        """Detect anomaly in metric value."""
        self._history.append(value)
        
        # Need minimum history for detection
        if len(self._history) < self._config.min_history_size:
            return None
        
        # Update baseline statistics
        self._update_baseline()
        
        # Calculate z-score
        if self._baseline_std and self._baseline_std > 0:
            z_score = abs(value - self._baseline_mean) / self._baseline_std
            
            if z_score > self._config.critical_threshold:
                return AnomalyResult(
                    type=AnomalyType.STATISTICAL,
                    severity=AnomalySeverity.CRITICAL,
                    confidence=min(z_score / self._config.critical_threshold, 1.0),
                    details={"z_score": z_score, "threshold": self._config.critical_threshold}
                )
            elif z_score > self._config.warning_threshold:
                return AnomalyResult(
                    type=AnomalyType.STATISTICAL,
                    severity=AnomalySeverity.WARNING,
                    confidence=min(z_score / self._config.warning_threshold, 1.0),
                    details={"z_score": z_score, "threshold": self._config.warning_threshold}
                )
        
        return None
    
    def _update_baseline(self) -> None:
        """Update baseline statistics from history."""
        if len(self._history) >= self._config.min_history_size:
            values = list(self._history)
            self._baseline_mean = statistics.mean(values)
            self._baseline_std = statistics.stdev(values) if len(values) > 1 else 0
```

## Alerting System

### Alert Rules

**Purpose**: Defines conditions that trigger alerts and notifications.

```python
class AlertRule:
    """Defines conditions for triggering alerts."""
    
    def __init__(self, rule_id: str, name: str, condition: AlertCondition,
                 severity: AlertSeverity, notification_channels: List[str]):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.severity = severity
        self.notification_channels = notification_channels
        self.enabled = True
    
    def evaluate(self, system_state: SystemState) -> Optional[Alert]:
        """Evaluate rule against system state."""
        if not self.enabled:
            return None
        
        if self.condition.is_met(system_state):
            return Alert(
                alert_id=str(uuid.uuid4()),
                rule_id=self.rule_id,
                system_id=system_state.system_id,
                severity=self.severity,
                message=self.condition.get_message(system_state),
                timestamp=datetime.utcnow(),
                notification_channels=self.notification_channels
            )
        
        return None

class ThresholdAlertCondition(AlertCondition):
    """Alert condition based on metric threshold."""
    
    def __init__(self, metric_name: str, operator: str, threshold: float):
        self.metric_name = metric_name
        self.operator = operator  # >, <, >=, <=, ==, !=
        self.threshold = threshold
    
    def is_met(self, system_state: SystemState) -> bool:
        """Check if condition is met."""
        if not system_state.has_metric(self.metric_name):
            return False
        
        value = system_state.get_metric_value(self.metric_name)
        
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 0.001
        elif self.operator == "!=":
            return abs(value - self.threshold) >= 0.001
        
        return False
    
    def get_message(self, system_state: SystemState) -> str:
        """Get alert message."""
        value = system_state.get_metric_value(self.metric_name)
        return f"{self.metric_name} is {value}, which {self.operator} {self.threshold}"
```

### Notification System

**Purpose**: Delivers alerts through various channels.

```python
class NotificationManager:
    """Manages alert notifications across multiple channels."""
    
    def __init__(self, config: NotificationConfig):
        self._config = config
        self._channels: Dict[str, NotificationChannel] = {}
        self._setup_channels()
    
    async def send_alert(self, alert: Alert) -> None:
        """Send alert through configured channels."""
        for channel_name in alert.notification_channels:
            if channel_name in self._channels:
                channel = self._channels[channel_name]
                try:
                    await channel.send_notification(alert)
                    logger.info(f"Alert {alert.alert_id} sent via {channel_name}")
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel_name}: {e}")
    
    def _setup_channels(self) -> None:
        """Set up notification channels."""
        for channel_config in self._config.channels:
            if channel_config.type == "email":
                self._channels[channel_config.name] = EmailNotificationChannel(channel_config)
            elif channel_config.type == "slack":
                self._channels[channel_config.name] = SlackNotificationChannel(channel_config)
            elif channel_config.type == "webhook":
                self._channels[channel_config.name] = WebhookNotificationChannel(channel_config)

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    async def send_notification(self, alert: Alert) -> None:
        """Send alert via email."""
        subject = f"POLARIS Alert: {alert.severity.value.upper()} - {alert.system_id}"
        
        body = f"""
        Alert Details:
        - System: {alert.system_id}
        - Severity: {alert.severity.value}
        - Message: {alert.message}
        - Time: {alert.timestamp.isoformat()}
        - Alert ID: {alert.alert_id}
        """
        
        # Send email using configured SMTP settings
        await self._send_email(
            to=self._config.recipients,
            subject=subject,
            body=body
        )
```

## Performance Metrics

### Collection Metrics

```python
# Prometheus metrics for monitoring system performance
collection_duration = Histogram(
    'polaris_metric_collection_duration_seconds',
    'Time spent collecting metrics',
    ['system_id', 'strategy']
)

collection_count = Counter(
    'polaris_metric_collection_total',
    'Total number of metric collections',
    ['system_id', 'strategy', 'status']
)

batch_collection_duration = Histogram(
    'polaris_batch_collection_duration_seconds',
    'Time spent on batch collections',
    ['batch_size']
)

batch_success_rate = Histogram(
    'polaris_batch_success_rate',
    'Success rate of batch collections',
    ['batch_size']
)

push_received_count = Counter(
    'polaris_push_metrics_received_total',
    'Total number of pushed metrics received',
    ['system_id']
)

push_error_count = Counter(
    'polaris_push_metrics_errors_total',
    'Total number of push metric errors',
    ['system_id']
)
```

### System Health Metrics

```python
system_health_score = Gauge(
    'polaris_system_health_score',
    'Current system health score (0-1)',
    ['system_id']
)

anomaly_detection_count = Counter(
    'polaris_anomalies_detected_total',
    'Total number of anomalies detected',
    ['system_id', 'metric_name', 'severity']
)

alert_count = Counter(
    'polaris_alerts_triggered_total',
    'Total number of alerts triggered',
    ['system_id', 'severity', 'rule_id']
)

notification_count = Counter(
    'polaris_notifications_sent_total',
    'Total number of notifications sent',
    ['channel', 'status']
)
```

## Configuration Examples

### Monitoring Configuration

```yaml
monitoring:
  # Collection strategies
  strategies:
    - name: "critical_systems"
      type: "pull"
      interval: 10  # seconds
      timeout: 5
      systems: ["database", "api_gateway"]
    
    - name: "standard_systems"
      type: "pull"
      interval: 30
      timeout: 10
      systems: ["web_servers", "cache"]
    
    - name: "batch_collection"
      type: "batch"
      interval: 60
      batch_size: 20
      timeout: 30
  
  # Health assessment
  health_assessment:
    rules:
      - metric: "cpu_usage"
        warning_threshold: 0.8
        critical_threshold: 0.95
      
      - metric: "memory_usage"
        warning_threshold: 0.85
        critical_threshold: 0.95
      
      - metric: "error_rate"
        warning_threshold: 0.01
        critical_threshold: 0.05
  
  # Anomaly detection
  anomaly_detection:
    enabled: true
    history_size: 100
    min_history_size: 20
    warning_threshold: 2.0  # z-score
    critical_threshold: 3.0  # z-score
  
  # Alerting
  alerting:
    rules:
      - name: "High CPU Usage"
        condition:
          metric: "cpu_usage"
          operator: ">="
          threshold: 0.9
        severity: "warning"
        channels: ["email", "slack"]
      
      - name: "Critical Memory Usage"
        condition:
          metric: "memory_usage"
          operator: ">="
          threshold: 0.95
        severity: "critical"
        channels: ["email", "slack", "pagerduty"]
    
    channels:
      - name: "email"
        type: "email"
        config:
          smtp_server: "smtp.company.com"
          recipients: ["ops@company.com"]
      
      - name: "slack"
        type: "slack"
        config:
          webhook_url: "https://hooks.slack.com/..."
          channel: "#alerts"
```

---

*Continue to [Adaptive Control](./14-adaptive-control.md) →*