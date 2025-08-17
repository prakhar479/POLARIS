"""
Query and response models for POLARIS Knowledge Base - Telemetry Focused.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class QueryType(str, Enum):
    """Types of queries supported by the knowledge base."""

    STRUCTURED = "structured"
    NATURAL_LANGUAGE = "natural_language"
    SEMANTIC = "semantic"
    METRIC_RANGE = "metric_range"
    TIME_SERIES = "time_series"


class KBDataType(str, Enum):
    """Types of data that can be stored in the knowledge base."""

    TELEMETRY_EVENT = "telemetry_event"
    METRIC_BATCH = "metric_batch"
    SYSTEM_STATE = "system_state"
    ALERT = "alert"
    CONFIGURATION = "configuration"
    GENERIC = "generic"


class KBQuery(BaseModel):
    """Knowledge base query model with telemetry-focused enhancements."""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_type: QueryType
    content: str = Field(
        default="",
        description="The query content (natural language or structured query string)",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured filters for precise data retrieval"
    )

    # Telemetry-specific query parameters
    metric_name: Optional[str] = Field(
        default=None, description="Specific metric name to query (e.g., 'cpu.usage')"
    )
    time_range: Optional[Dict[str, str]] = Field(
        default=None,
        description="Time range for queries (start_time, end_time in ISO format)",
    )
    sources: Optional[List[str]] = Field(
        default=None, description="List of source systems to filter by"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None, description="Tag filters for telemetry data"
    )

    # Generic query parameters
    data_types: Optional[List[KBDataType]] = Field(
        default=None, description="Types of data to include in search"
    )
    limit: Optional[int] = Field(
        default=100, description="Maximum number of results to return"
    )
    offset: Optional[int] = Field(default=0, description="Offset for pagination")

    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional query context"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def validate_query_requirements(self):
        """Validate that query has sufficient parameters."""
        has_content = self.content and self.content.strip()
        has_filters = self.filters and len(self.filters) > 0
        has_metric = self.metric_name is not None
        has_time_range = self.time_range is not None

        if not any([has_content, has_filters, has_metric, has_time_range]):
            raise ValueError(
                "Query must have at least one of: content, filters, metric_name, or time_range"
            )
        return self


class KBEntry(BaseModel):
    """
    Generic knowledge base entry optimized for telemetry data storage.
    """

    entry_id: str = Field(..., description="Unique entry identifier")
    data_type: KBDataType = Field(
        default=KBDataType.GENERIC, description="Type of data stored in this entry"
    )

    # Core data fields
    content: Dict[str, Any] = Field(
        ..., description="Primary data content - flexible structure"
    )

    # Telemetry-optimized fields
    metric_name: Optional[str] = Field(
        default=None,
        description="Metric name for telemetry events (extracted from content)",
    )
    metric_value: Optional[Union[float, int, bool, str]] = Field(
        default=None, description="Primary metric value for quick access"
    )
    source: Optional[str] = Field(
        default=None, description="Source system that generated this data"
    )

    # Generic metadata
    tags: Optional[List[str]] = Field(
        default=None, description="Searchable tags for categorization"
    )
    labels: Optional[Dict[str, str]] = Field(
        default=None, description="Key-value labels for filtering (telemetry-style)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional structured metadata"
    )

    # Timestamps
    event_timestamp: Optional[str] = Field(
        default=None,
        description="Timestamp when the actual event occurred (for telemetry)",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("metric_name", mode="before")
    def normalize_metric_name(cls, v):
        """Normalize metric name if provided."""
        if v:
            return v.lower().replace("_", ".")
        return v

    @model_validator(mode="after")
    def extract_telemetry_fields(self):
        """Auto-extract telemetry fields from content when applicable."""
        if self.data_type == KBDataType.TELEMETRY_EVENT and self.content:
            # Extract metric name if not set
            if not self.metric_name and "name" in self.content:
                self.metric_name = str(self.content["name"]).lower().replace("_", ".")

            # Extract metric value if not set
            if self.metric_value is None and "value" in self.content:
                self.metric_value = self.content["value"]

            # Extract source if not set
            if not self.source and "source" in self.content:
                self.source = self.content["source"]

            # Extract event timestamp if not set
            if not self.event_timestamp and "timestamp" in self.content:
                self.event_timestamp = self.content["timestamp"]

            # Extract labels from tags in content
            if (
                not self.labels
                and "tags" in self.content
                and isinstance(self.content["tags"], dict)
            ):
                self.labels = self.content["tags"]

        return self

    def to_telemetry_event(self) -> Optional[Dict[str, Any]]:
        """Convert back to telemetry event format if applicable."""
        if self.data_type != KBDataType.TELEMETRY_EVENT:
            return None

        event = {
            "name": self.metric_name or self.content.get("name"),
            "value": self.metric_value or self.content.get("value"),
            "timestamp": self.event_timestamp or self.content.get("timestamp"),
            "source": self.source or self.content.get("source", "unknown"),
        }

        # Add optional fields
        if "unit" in self.content:
            event["unit"] = self.content["unit"]
        if self.labels:
            event["tags"] = self.labels
        if "metadata" in self.content:
            event["metadata"] = self.content["metadata"]

        return event


class KBResponse(BaseModel):
    """Enhanced knowledge base query response with telemetry insights."""

    query_id: str
    success: bool
    results: List[KBEntry] = Field(default_factory=list)
    total_results: int = Field(default=0)
    message: Optional[str] = Field(default=None)

    # Enhanced metadata for telemetry queries
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    processing_time_ms: Optional[float] = Field(default=None)

    # Telemetry-specific response data
    metric_summary: Optional[Dict[str, Any]] = Field(
        default=None, description="Summary statistics for metric queries"
    )
    time_range_covered: Optional[Dict[str, str]] = Field(
        default=None, description="Actual time range of returned data"
    )
    sources_found: Optional[List[str]] = Field(
        default=None, description="List of sources in the results"
    )
    data_types_found: Optional[List[KBDataType]] = Field(
        default=None, description="Types of data found in results"
    )

    @model_validator(mode="after")
    def compute_response_metadata(self):
        """Compute additional metadata from results."""
        if self.results:
            # Extract unique sources
            sources = set()
            data_types = set()
            timestamps = []

            for entry in self.results:
                if entry.source:
                    sources.add(entry.source)
                data_types.add(entry.data_type)
                if entry.event_timestamp:
                    timestamps.append(entry.event_timestamp)

            self.sources_found = sorted(list(sources))
            self.data_types_found = sorted(list(data_types))

            # Compute time range if timestamps available
            if timestamps:
                self.time_range_covered = {
                    "start_time": min(timestamps),
                    "end_time": max(timestamps),
                }

            # Compute metric summary for telemetry events
            telemetry_entries = [
                e for e in self.results if e.data_type == KBDataType.TELEMETRY_EVENT
            ]
            if telemetry_entries:
                metric_values = []
                metric_names = set()

                for entry in telemetry_entries:
                    if entry.metric_value is not None and isinstance(
                        entry.metric_value, (int, float)
                    ):
                        metric_values.append(float(entry.metric_value))
                    if entry.metric_name:
                        metric_names.add(entry.metric_name)

                if metric_values:
                    self.metric_summary = {
                        "count": len(metric_values),
                        "min": min(metric_values),
                        "max": max(metric_values),
                        "avg": sum(metric_values) / len(metric_values),
                        "unique_metrics": sorted(list(metric_names)),
                    }

        return self


class TelemetryQueryBuilder:
    """Helper class to build telemetry-focused queries."""

    @staticmethod
    def metric_range_query(
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        time_range: Optional[Dict[str, str]] = None,
        sources: Optional[List[str]] = None,
    ) -> KBQuery:
        """Build a metric range query."""
        filters = {"data_type": KBDataType.TELEMETRY_EVENT}

        if min_value is not None:
            filters["metric_value__gte"] = min_value
        if max_value is not None:
            filters["metric_value__lte"] = max_value

        return KBQuery(
            query_type=QueryType.METRIC_RANGE,
            metric_name=metric_name,
            time_range=time_range,
            sources=sources,
            filters=filters,
        )

    @staticmethod
    def time_series_query(
        metric_name: str,
        start_time: str,
        end_time: str,
        sources: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> KBQuery:
        """Build a time series query."""
        return KBQuery(
            query_type=QueryType.TIME_SERIES,
            metric_name=metric_name,
            time_range={"start_time": start_time, "end_time": end_time},
            sources=sources,
            tags=tags,
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
        )

    @staticmethod
    def natural_language_query(
        content: str, data_types: Optional[List[KBDataType]] = None
    ) -> KBQuery:
        """Build a natural language query."""
        return KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            content=content,
            data_types=data_types or [KBDataType.TELEMETRY_EVENT],
        )
