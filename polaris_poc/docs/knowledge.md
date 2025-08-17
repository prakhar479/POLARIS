```python
"""
Complete examples of how to query the POLARIS Knowledge Base
"""

from polaris.common.knowledge_base import get_knowledge_base, KBQuery, QueryType

# Initialize knowledge base
kb = get_knowledge_base()

# ==== METHOD 1: Simple String Query (Natural Language) ====
# This is the easiest way - just pass a string
response = kb.query("CPU issues")
print(f"Found {response.total_results} results")
for result in response.results:
    print(f"- {result.entry_id}: {result.content}")

# ==== METHOD 2: Structured Query with Filters ====
# Use filters for exact matches
structured_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",  # Can be empty for filter-only queries
    filters={
        "tags": ["incident", "web_service"],  # Must have these tags
        "severity": "high"  # content or metadata must have severity=high
    }
)
response = kb.query(structured_query)

# ==== METHOD 3: Content-based Text Search ====
# Search within the actual content of entries
text_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="database connection timeout",  # Searches in content text
    filters=None
)
response = kb.query(text_query)

# ==== METHOD 4: Combined Filters + Text ====
# Use both filters and text search
combined_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="scaling",  # Text to search for
    filters={
        "tags": ["adaptation"]  # Must be adaptation-related
    }
)
response = kb.query(combined_query)

# ==== METHOD 5: Convenience Methods ====
# Use built-in helper methods for common queries

# Find similar incidents
incidents = kb.get_similar_incidents(
    incident_type="high_cpu",
    system="web_service"
)

# Get adaptation history
adaptations = kb.get_adaptation_history(
    action_type="scale_out",
    system="web_service"
)

# ==== PRACTICAL EXAMPLES ====

# Example 1: Store some data first
kb.store_incident(
    incident_type="database_timeout",
    description="Database queries timing out after 30 seconds",
    resolution="Increased connection pool size and added query optimization",
    system="api_service",
    severity="high"
)

kb.store_adaptation_outcome(
    action_type="scale_out",
    outcome="Successfully handled traffic spike",
    metrics={"instances_before": 2, "instances_after": 5},
    system="api_service",
    success=True
)

# Example 2: Query by severity
high_severity_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={"tags": ["high"]}  # All high severity items
)
high_issues = kb.query(high_severity_query)

# Example 3: Query by system
api_issues = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={"tags": ["api_service"]}  # All api_service related
)
response = kb.query(api_issues)

# Example 4: Search for specific error patterns
error_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="timeout",  # Text search for "timeout"
    filters={"tags": ["incident"]}  # Only incidents
)
timeout_incidents = kb.query(error_query)

# Example 5: Natural language queries (currently simple text search)
nl_responses = [
    kb.query("show me all scaling actions"),
    kb.query("what happened with database last week"),
    kb.query("successful adaptations"),
    kb.query("high severity incidents")
]

# ==== WORKING WITH QUERY RESULTS ====
response = kb.query("database")

if response.success:
    print(f"Query took {response.processing_time_ms:.2f}ms")
    print(f"Found {response.total_results} results")

    for entry in response.results:
        print(f"\nEntry ID: {entry.entry_id}")
        print(f"Created: {entry.created_at}")
        print(f"Tags: {entry.tags}")
        print(f"Content: {entry.content}")
        if entry.metadata:
            print(f"Metadata: {entry.metadata}")
else:
    print(f"Query failed: {response.message}")

# ==== ADVANCED FILTERING ====

# Multiple tag filtering (AND logic - must have ALL tags)
multi_tag_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={"tags": ["incident", "database", "high"]}  # Must have all three
)

# Specific content field matching
content_filter_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={
        "incident_type": "database_timeout",  # Exact match in content
        "system": "api_service"
    }
)

# Metadata filtering
metadata_query = KBQuery(
    query_type=QueryType.STRUCTURED,
    content="",
    filters={"category": "adaptation"}  # Matches metadata field
)

# ==== INTEGRATION EXAMPLE: In Monitor Adapter ====
"""
# In your monitor adapter when detecting anomaly:
class MonitorAdapter(BaseAdapter):
    def __init__(self, ...):
        super().__init__(...)
        self.kb = get_knowledge_base(self.logger)

    async def _handle_cpu_spike(self, cpu_value):
        # Store the incident
        incident_id = self.kb.store_incident(
            incident_type="cpu_spike",
            description=f"CPU usage reached {cpu_value}%",
            resolution="Pending investigation",
            system=self.plugin_config["system_name"],
            severity="high" if cpu_value > 90 else "medium"
        )

        # Query for similar past incidents
        similar_incidents = self.kb.get_similar_incidents("cpu_spike", self.plugin_config["system_name"])

        if similar_incidents:
            self.logger.info(f"Found {len(similar_incidents)} similar incidents")
            # Could use this info to suggest actions
            for incident in similar_incidents:
                if incident.content.get("resolution") != "Pending investigation":
                    self.logger.info(f"Past resolution: {incident.content['resolution']}")
"""

# ==== INTEGRATION EXAMPLE: In Execution Adapter ====
"""
# In your execution adapter after action completion:
async def execute_action(self, action: ControlAction) -> ExecutionResult:
    # ... existing execution logic ...

    # Store outcome
    self.kb.store_adaptation_outcome(
        action_type=action.action_type,
        outcome=result.message,
        metrics={
            "duration_sec": result.duration_sec,
            "success_rate": 1.0 if result.success else 0.0
        },
        system=self.plugin_config["system_name"],
        success=result.success
    )

    # Query for similar action history
    history = self.kb.get_adaptation_history(action.action_type, self.plugin_config["system_name"])
    success_rate = sum(1 for h in history if h.content.get("success")) / len(history) if history else 0

    self.logger.info(f"Action {action.action_type} historical success rate: {success_rate:.2%}")

    return result
"""

print("Knowledge base query examples completed!")
```
