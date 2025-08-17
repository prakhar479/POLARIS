# POLARIS Knowledge Base - Enhanced Telemetry-Focused Implementation

## Data Models

### KBEntry - Enhanced Telemetry Entry

```python
from polaris.common.query_models import KBEntry, KBDataType

# Telemetry-optimized entry with automatic field extraction
entry = KBEntry(
    entry_id="telemetry-cpu-001",
    data_type=KBDataType.TELEMETRY_EVENT,
    content={
        "name": "cpu.usage",
        "value": 85.5,
        "timestamp": "2025-08-17T10:00:00Z",
        "source": "server-001",
        "tags": {"environment": "production", "service": "web"},
        "unit": "percent"
    },
    # Fields automatically extracted from content:
    # metric_name, metric_value, source, event_timestamp, labels
)
```

### KBQuery - Advanced Query Types

```python
from polaris.common.query_models import KBQuery, QueryType, TelemetryQueryBuilder

# 1. Natural Language Query
query = TelemetryQueryBuilder.natural_language_query("high CPU usage")

# 2. Structured Query with Advanced Filters
query = KBQuery(
    query_type=QueryType.STRUCTURED,
    filters={
        "metric_value__gt": 80,           # CPU > 80%
        "source__contains": "web",        # Source contains "web"
        "labels.environment": "production" # Environment label
    }
)

# 3. Metric Range Query
query = TelemetryQueryBuilder.metric_range_query(
    metric_name="cpu.usage",
    min_value=70.0,
    max_value=95.0,
    time_range={"start_time": "2025-08-17T09:00:00Z", "end_time": "2025-08-17T11:00:00Z"}
)

# 4. Time Series Query
query = TelemetryQueryBuilder.time_series_query(
    metric_name="response.time",
    start_time="2025-08-17T10:00:00Z",
    end_time="2025-08-17T11:00:00Z",
    sources=["web-server-01", "web-server-02"]
)
```

## Enhanced Query Capabilities

### Advanced Filter Operators

The KB now supports sophisticated filtering with operator suffixes:

```python
# Numeric Comparisons
filters = {
    "cpu_usage__gt": 80,        # Greater than
    "cpu_usage__gte": 75,       # Greater than or equal
    "memory_usage__lt": 90,     # Less than
    "memory_usage__lte": 85,    # Less than or equal
    "response_time__ne": 1000   # Not equal
}

# String Operations
filters = {
    "hostname__contains": "web",     # Contains substring
    "service__startswith": "api",    # Starts with
    "process__endswith": ".exe",     # Ends with
    "error_msg__regex": "ERROR|FATAL" # Regex pattern
}

# List Operations
filters = {
    "status__in": ["warning", "critical"],      # Value in list
    "tier__not_in": ["development", "staging"]  # Value not in list
}

# Nested Field Access
filters = {
    "metrics.cpu.cores__gt": 8,              # Nested object access
    "config.database.pool_size__lt": 10,     # Deep nesting
    "labels.environment": "production"        # Label access
}
```

### Natural Language Search Enhancement

The natural language search now provides comprehensive content matching:

```python
query = TelemetryQueryBuilder.natural_language_query("cpu usage high performance")

# Searches across:
# - Content fields and values
# - Metric names
# - Tags and labels
# - Source identifiers
# - All query terms must be found (AND logic)
```

## Storage Implementation

### Enhanced InMemoryKBStorage

```python
from polaris.common.kb_storage import InMemoryKBStorage

storage = InMemoryKBStorage()

# Automatic telemetry field extraction and indexing
storage.store_telemetry_event({
    "name": "memory.usage",
    "value": 72.3,
    "timestamp": "2025-08-17T10:01:00Z",
    "source": "server-002",
    "tags": {"environment": "production"}
})

# Batch telemetry processing
batch_result = storage.store_telemetry_batch({
    "events": [
        {"name": "cpu.usage", "value": 85.0, "timestamp": "2025-08-17T10:00:00Z"},
        {"name": "memory.usage", "value": 70.5, "timestamp": "2025-08-17T10:01:00Z"}
    ]
})
print(f"Stored: {batch_result['stored']}, Failed: {batch_result['failed']}")
```

### Smart Indexing System

The storage automatically maintains multiple indexes for fast querying:

- **Metric Index**: Metric names → entry IDs
- **Source Index**: Source systems → entry IDs
- **Tag Index**: Tags → entry IDs
- **Label Index**: Label key-value pairs → entry IDs
- **Data Type Index**: Data types → entry IDs
- **Time Index**: Chronologically sorted timestamps

### Telemetry-Specific Methods

```python
# Get telemetry metrics with filtering
metrics = storage.get_telemetry_metrics(
    metric_name="cpu.usage",
    start_time="2025-08-17T10:00:00Z",
    end_time="2025-08-17T11:00:00Z"
)

# Get metric summary statistics
summary = storage.get_metric_summary("cpu.usage")
print(f"Average: {summary['value_statistics']['avg']}")
print(f"Min/Max: {summary['value_statistics']['min']}/{summary['value_statistics']['max']}")
print(f"Sources: {summary['unique_sources']}")

# Get all available metrics and sources
all_metrics = storage.get_all_metrics()
all_sources = storage.get_all_sources()
```

## Integration Examples

### Monitor Adapter Integration

```python
class EnhancedMonitorAdapter(BaseAdapter):
    def __init__(self, ...):
        super().__init__(...)
        self.kb_service = EnhancedKnowledgeBaseService()

    async def _process_metrics(self, metrics):
        # Store current metrics
        for metric in metrics:
            self.kb_service.add_telemetry_event({
                "name": metric.name,
                "value": metric.value,
                "timestamp": metric.timestamp,
                "source": self.plugin_config["system_name"],
                "tags": {"adapter": "monitor", "plugin": self.plugin_config["system_name"]}
            })

        # Query for similar historical situations
        similar_load = self.kb_service.process_query(KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={
                "metric_name": "cpu.usage",
                "metric_value__gte": metrics.cpu_usage - 10,
                "metric_value__lte": metrics.cpu_usage + 10,
                "source": self.plugin_config["system_name"]
            }
        ))

        # Analyze patterns for predictive insights
        if similar_load.total_results > 5:
            # Use historical data for predictions
            pass
```

### Knowledge-Based Decision Making

```python
class IntelligentDecisionEngine:
    def __init__(self, kb_service):
        self.kb_service = kb_service

    def should_scale_out(self, current_metrics) -> bool:
        # Query historical scaling decisions
        past_decisions = self.kb_service.process_query(KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={
                "metric_name": "cpu.usage",
                "metric_value__gte": current_metrics.cpu_usage - 10,
                "metric_value__lte": current_metrics.cpu_usage + 10,
                "action_taken": "scale_out",
                "labels.outcome": "success"
            }
        ))

        success_rate = past_decisions.total_results / max(1, self._get_total_decisions())
        return success_rate > 0.7 and current_metrics.cpu_usage > 80

    def analyze_system_health(self):
        # Complex multi-metric analysis
        health_query = self.kb_service.process_query(KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={
                "metric_value__gt": 90,  # High resource usage
                "labels.severity__in": ["warning", "critical"],
                "timestamp__gte": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ))

        return self._compute_health_score(health_query.results)
```

### Digital Twin Integration

```python
class DigitalTwinKBIntegration:
    def __init__(self, kb_service):
        self.kb_service = kb_service

    async def process_knowledge_event(self, event: KnowledgeEvent):
        if event.event_type == "telemetry":
            # Store telemetry in KB for historical analysis
            self.kb_service.add_telemetry_event(event.data.to_dict())

        elif event.event_type == "execution_status":
            # Store execution results for pattern analysis
            self.kb_service.add_entry(KBEntry(
                entry_id=f"execution-{event.event_id}",
                data_type=KBDataType.SYSTEM_STATE,
                content=event.data.to_dict(),
                tags=["execution", "result"]
            ))

    async def query_historical_patterns(self, query_content: str):
        # Use natural language query for flexible analysis
        response = self.kb_service.process_query(
            TelemetryQueryBuilder.natural_language_query(query_content)
        )

        return self._analyze_patterns(response.results)
```

## Performance Optimization

### Indexing Strategy

The KB uses intelligent indexing to optimize different query patterns:

```python
# Time-based queries use sorted time index
time_filtered = storage._filter_by_time_range("2025-08-17T10:00:00Z", "2025-08-17T11:00:00Z")

# Metric queries use metric index
metric_entries = storage._metric_index.get("cpu.usage", set())

# Complex queries combine multiple indexes
candidate_ids = storage._get_candidate_ids_from_indexes(query)
```

### Memory Management

```python
# Get comprehensive storage statistics
stats = storage.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Memory usage: {stats['memory_usage_estimate_kb']} KB")
print(f"Index efficiency: {stats['indexes']}")

# Data type breakdown
for data_type, count in stats['data_type_breakdown'].items():
    print(f"{data_type}: {count} entries")
```

### Query Performance

```python
# Optimized range queries
results = storage.search_by_metric_range("cpu.usage", min_value=80, max_value=95)

# Efficient time series queries
time_series = storage.get_telemetry_metrics(
    metric_name="response.time",
    start_time="2025-08-17T10:00:00Z",
    end_time="2025-08-17T11:00:00Z"
)

# Fast content search with improved matching
content_results = storage.search_by_content("high cpu performance issues")
```

## Testing and Validation

### Unit Testing Examples

```python
import pytest
from polaris.common.kb_storage import InMemoryKBStorage
from polaris.common.query_models import KBQuery, QueryType, KBDataType

def test_telemetry_storage_and_query():
    storage = InMemoryKBStorage()

    # Store telemetry event
    result = storage.store_telemetry_event({
        "name": "cpu.usage",
        "value": 85.5,
        "timestamp": "2025-08-17T10:00:00Z",
        "source": "test-server"
    })
    assert result is True

    # Query by metric range
    query = KBQuery(
        query_type=QueryType.STRUCTURED,
        filters={"metric_value__gt": 80}
    )
    response = storage.query(query)
    assert response.success is True
    assert response.total_results == 1

def test_natural_language_search():
    storage = InMemoryKBStorage()

    # Store test data
    storage.store_telemetry_event({
        "name": "cpu.usage",
        "value": 90.0,
        "source": "web-server"
    })

    # Natural language query
    query = TelemetryQueryBuilder.natural_language_query("cpu usage web")
    response = storage.query(query)
    assert response.total_results >= 1
```

### Integration Testing

```python
def test_kb_integration_with_adapters():
    """Test KB integration with POLARIS adapters."""
    kb_service = EnhancedKnowledgeBaseService()

    # Simulate monitor adapter data
    telemetry_batch = {
        "events": [
            {"name": "cpu.usage", "value": 85.0, "source": "test-server"},
            {"name": "memory.usage", "value": 70.0, "source": "test-server"}
        ]
    }

    result = kb_service.add_telemetry_batch(telemetry_batch)
    assert result["stored"] == 2

    # Query the stored data
    query = KBQuery(
        query_type=QueryType.STRUCTURED,
        filters={"source": "test-server"}
    )
    response = kb_service.process_query(query)
    assert response.total_results == 2
```

## Configuration

### KB Service Configuration

```yaml
# In polaris_config.yaml
knowledge_base:
  storage:
    implementation: "memory" # "memory", "redis", "postgresql" (future)
    max_entries: 100000
    cleanup_interval_sec: 3600

  telemetry:
    auto_extract_fields: true
    normalize_metric_names: true
    index_all_fields: true

  performance:
    batch_size: 100
    query_timeout_sec: 30
    max_concurrent_queries: 10

  analytics:
    enable_summaries: true
    retention_days: 30
    compute_statistics: true
```

### Monitor Adapter KB Integration

```yaml
# In plugin config
monitoring:
  knowledge_base:
    enabled: true
    store_all_metrics: true
    store_derived_metrics: true
    tags:
      - "monitoring"
      - "telemetry"
```
