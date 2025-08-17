```python
"""
Complete examples of how to query the POLARIS Knowledge Base

Updated with latest enhancements including:
- Filter-only queries (empty content with filters)
- Advanced metric search operators
- Nested field access
- Range queries and convenience methods
"""

from polaris.common.query_models import KBQuery, QueryType, KBEntry
from polaris.common.kb_storage import InMemoryKBStorage

# Initialize knowledge base storage
storage = InMemoryKBStorage()

# ==== METHOD 1: Simple String Query (Natural Language) ====
# This is the easiest way - just pass a string
query = KBQuery(
    query_type=QueryType.NATURAL_LANGUAGE,
    content="CPU issues"
)
# Process with your KB service
print("Natural language query for CPU issues")

# ==== METHOD 2: Filter-Only Queries (NEW!) ====
# You can now query with EMPTY content if you provide filters
filter_only_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",  # Empty content is now allowed!
    filters={
        "severity": "high",
        "tags": ["incident", "web_service"]
    }
)
results = storage.search_by_filters(filter_only_query.filters)
print(f"Filter-only query found {len(results)} results")

# ==== METHOD 3: Exact Metric Searches ====
# Search for exact metric values
exact_metric_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "cpu_usage": 85.5,           # Exact CPU usage
        "memory_usage": 76.2,        # Exact memory usage
        "status": "critical",        # Exact status
        "node_id": "server-001"      # Exact identifier
    }
)
results = storage.search_by_filters(exact_metric_query.filters)

# ==== METHOD 4: Advanced Metric Operators (NEW!) ====
# Use advanced operators for sophisticated queries

# Numeric comparisons
numeric_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "cpu_usage__gt": 80,         # CPU > 80%
        "cpu_usage__lte": 95,        # CPU <= 95%
        "memory_usage__gte": 70,     # Memory >= 70%
        "disk_usage__lt": 90,        # Disk < 90%
        "response_time__ne": 1000    # Response time != 1000ms
    }
)

# String operations
string_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "hostname__contains": "web",      # Hostname contains "web"
        "service__startswith": "api",     # Service starts with "api"
        "process__endswith": ".exe",      # Process ends with ".exe"
        "error_msg__regex": "ERROR|FATAL", # Regex pattern matching
        "status__ne": "ok"                # Status is not "ok"
    }
)

# List operations
list_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "status__in": ["warning", "critical", "error"],    # Status in list
        "tier__not_in": ["development", "staging"],        # Tier not in list
        "tags": ["production", "database"]                 # Has any of these tags
    }
)

# ==== METHOD 5: Nested Field Access (NEW!) ====
# Access deep object properties using dot notation
nested_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "metrics.cpu.cores__gt": 8,              # CPU cores > 8
        "metrics.memory.total": 16384,           # Exact memory total
        "metrics.network.bytes_in__gte": 1000000, # Network traffic >= 1MB
        "config.database.pool_size__lt": 10,     # DB pool size < 10
        "status.health.overall": "healthy"       # Nested status check
    }
)

# ==== METHOD 6: Range Queries with Convenience Method (NEW!) ====
# Use the convenient range search method
cpu_range_results = storage.search_by_metric_range(
    "cpu_usage",
    min_value=40,    # CPU >= 40%
    max_value=90     # CPU <= 90%
)

memory_min_results = storage.search_by_metric_range(
    "memory_usage",
    min_value=70     # Memory >= 70% (no max limit)
)

disk_max_results = storage.search_by_metric_range(
    "disk_usage",
    max_value=50     # Disk <= 50% (no min limit)
)

# ==== METHOD 7: Complex Combined Queries (ENHANCED!) ====
# Combine multiple advanced filters
complex_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",  # Filter-only query
    filters={
        # Resource thresholds
        "cpu_usage__gt": 80,
        "memory_usage__gte": 70,
        "disk_usage__lt": 95,

        # Time-based filtering
        "uptime__gte": 86400,  # Uptime >= 24 hours

        # String matching
        "hostname__contains": "prod",
        "service__startswith": "web",

        # List operations
        "status__in": ["warning", "critical"],
        "tier__not_in": ["development"],

        # Nested metrics
        "metrics.network.connections__gt": 1000,
        "metrics.database.query_time__lt": 100,

        # Exact matches
        "datacenter": "us-east-1",
        "tags": ["production", "monitored"]
    }
)

# ==== PRACTICAL EXAMPLES ====

# Example 1: Store metric data with nested structure
server_entry = KBEntry(
    entry_id="server-web-001",
    content={
        "hostname": "web-server-001",
        "cpu_usage": 85.5,
        "memory_usage": 76.2,
        "disk_usage": 45,
        "status": "warning",
        "uptime": 172800,  # 48 hours
        "metrics": {
            "cpu": {"usage": 85.5, "cores": 8, "load_avg": 2.3},
            "memory": {"total": 16384, "used": 12488, "free": 3896},
            "network": {
                "bytes_in": 1024000,
                "bytes_out": 512000,
                "connections": 1250,
                "errors": 0
            },
            "database": {
                "query_time": 45,
                "connections": 25,
                "slow_queries": 2
            }
        },
        "config": {
            "database": {"pool_size": 20, "timeout": 30},
            "cache": {"size": 1024, "ttl": 300}
        }
    },
    tags=["server", "web", "production", "monitored"],
    metadata={
        "datacenter": "us-east-1",
        "tier": "frontend",
        "deployed_at": "2025-08-15T10:00:00Z"
    }
)
storage.store(server_entry)

# Example 2: Find high-resource usage servers
high_usage_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "cpu_usage__gt": 80,
        "memory_usage__gt": 70,
        "status__ne": "ok",
        "tags": ["production"]
    }
)
high_usage_servers = storage.search_by_filters(high_usage_query.filters)

# Example 3: Database performance monitoring
db_perf_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "metrics.database.query_time__gt": 100,  # Slow queries
        "metrics.database.connections__gte": 50,  # High connection count
        "tags": ["database"],
        "datacenter": "us-east-1"
    }
)
slow_db_servers = storage.search_by_filters(db_perf_query.filters)

# Example 4: Network traffic analysis
network_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "metrics.network.bytes_in__gt": 5000000,    # > 5MB incoming
        "metrics.network.connections__gt": 1000,     # > 1000 connections
        "metrics.network.errors__gt": 0,             # Has network errors
        "hostname__contains": "web"                   # Web servers only
    }
)
high_traffic_servers = storage.search_by_filters(network_query.filters)

# Example 5: Configuration audit
config_audit_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "config.database.pool_size__lt": 10,     # Small DB pool
        "config.cache.ttl__gt": 600,             # Long cache TTL
        "tier__in": ["frontend", "backend"],     # Specific tiers
        "uptime__lt": 86400                      # Recently restarted
    }
)
config_issues = storage.search_by_filters(config_audit_query.filters)

# ==== WORKING WITH QUERY RESULTS ====
def analyze_results(results: List[KBEntry], query_description: str):
    """Analyze and display query results."""
    print(f"\n{query_description}")
    print(f"Found {len(results)} matching entries")

    for entry in results:
        print(f"\nEntry ID: {entry.entry_id}")
        print(f"Hostname: {entry.content.get('hostname', 'N/A')}")
        print(f"CPU: {entry.content.get('cpu_usage', 'N/A')}%")
        print(f"Memory: {entry.content.get('memory_usage', 'N/A')}%")
        print(f"Status: {entry.content.get('status', 'N/A')}")
        print(f"Tags: {', '.join(entry.tags) if entry.tags else 'None'}")

        # Show nested metrics if available
        if 'metrics' in entry.content:
            metrics = entry.content['metrics']
            if 'network' in metrics:
                net = metrics['network']
                print(f"Network: {net.get('bytes_in', 0)} bytes in, {net.get('connections', 0)} connections")

# Usage examples
analyze_results(high_usage_servers, "High Resource Usage Servers")
analyze_results(slow_db_servers, "Servers with Database Performance Issues")
analyze_results(high_traffic_servers, "High Network Traffic Servers")

# ==== INTEGRATION EXAMPLE: Enhanced Monitor Adapter ====
"""
# In your monitor adapter with advanced querying:
class EnhancedMonitorAdapter(BaseAdapter):
    def __init__(self, ...):
        super().__init__(...)
        self.storage = InMemoryKBStorage(self.logger)

    async def _analyze_system_health(self, metrics):
        # Store current metrics
        entry = KBEntry(
            entry_id=f"metrics-{datetime.now().isoformat()}",
            content={
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "response_time": metrics.response_time,
                "error_rate": metrics.error_rate,
                "metrics": {
                    "detailed": metrics.detailed_metrics,
                    "timestamps": metrics.timestamps
                }
            },
            tags=["metrics", "monitoring", self.plugin_config["system_name"]],
            metadata={"source": "monitor", "timestamp": datetime.now().isoformat()}
        )
        self.storage.store(entry)

        # Query for similar high-load situations
        similar_load = self.storage.search_by_filters({
            "cpu_usage__gte": metrics.cpu_usage - 5,  # Within 5% CPU
            "memory_usage__gte": metrics.memory_usage - 10,  # Within 10% memory
            "tags": [self.plugin_config["system_name"]],
            "error_rate__lt": 0.1  # Low error rate (successful scenarios)
        })

        # Analyze patterns for predictive insights
        if len(similar_load) > 5:  # Enough historical data
            avg_response_time = sum(
                entry.content.get('response_time', 0) for entry in similar_load
            ) / len(similar_load)

            if metrics.response_time > avg_response_time * 1.5:
                self.logger.warning(
                    f"Response time {metrics.response_time}ms significantly higher "
                    f"than historical average {avg_response_time:.2f}ms"
                )
"""

# ==== INTEGRATION EXAMPLE: Knowledge-Based Decision Making ====
"""
class IntelligentDecisionEngine:
    def __init__(self, storage: InMemoryKBStorage):
        self.storage = storage

    def should_scale_out(self, current_metrics) -> bool:
        # Query for past scaling decisions with similar metrics
        similar_situations = self.storage.search_by_filters({
            "cpu_usage__gte": current_metrics.cpu_usage - 10,
            "cpu_usage__lte": current_metrics.cpu_usage + 10,
            "memory_usage__gte": current_metrics.memory_usage - 15,
            "memory_usage__lte": current_metrics.memory_usage + 15,
            "tags": ["scaling_decision"],
            "action_taken": "scale_out"
        })

        # Check success rate of past decisions
        successful_scales = self.storage.search_by_filters({
            "tags": ["scaling_decision"],
            "action_taken": "scale_out",
            "outcome": "success",
            "metrics.cpu_before__gte": current_metrics.cpu_usage - 10
        })

        success_rate = len(successful_scales) / len(similar_situations) if similar_situations else 0

        # Decision logic based on historical success
        return success_rate > 0.7 and current_metrics.cpu_usage > 80

    def get_optimal_scale_count(self, current_load) -> int:
        # Find successful scaling actions with similar load
        successful_scales = self.storage.search_by_filters({
            "metrics.load_before__gte": current_load * 0.9,
            "metrics.load_before__lte": current_load * 1.1,
            "outcome": "success",
            "tags": ["scaling_action"]
        })

        if successful_scales:
            # Return most common successful scale count
            scale_counts = [
                entry.content.get('instances_added', 1)
                for entry in successful_scales
            ]
            return max(set(scale_counts), key=scale_counts.count)

        return 1  # Default conservative scaling
"""

print("Enhanced Knowledge Base query examples with advanced metric search completed!")
```
