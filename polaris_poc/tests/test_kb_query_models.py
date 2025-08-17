"""
Unit tests for POLARIS Knowledge Base Query Models - Enhanced Telemetry Version.

Tests the enhanced query models, responses, and data structures used
in the POLARIS Knowledge Base system with telemetry optimization.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import uuid
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.query_models import (
    QueryType,
    KBDataType,
    KBQuery,
    KBEntry,
    KBResponse,
    TelemetryQueryBuilder,
)


class TestQueryType:
    """Test cases for QueryType enum."""

    def test_query_type_values(self):
        """Test QueryType enum values."""
        assert QueryType.STRUCTURED == "structured"
        assert QueryType.NATURAL_LANGUAGE == "natural_language"
        assert QueryType.SEMANTIC == "semantic"
        assert QueryType.METRIC_RANGE == "metric_range"
        assert QueryType.TIME_SERIES == "time_series"

    def test_query_type_membership(self):
        """Test QueryType membership checks."""
        assert QueryType.STRUCTURED in QueryType
        assert QueryType.NATURAL_LANGUAGE in QueryType
        assert QueryType.SEMANTIC in QueryType
        assert QueryType.METRIC_RANGE in QueryType
        assert QueryType.TIME_SERIES in QueryType

        # Test values
        assert QueryType.STRUCTURED.value == "structured"
        assert QueryType.NATURAL_LANGUAGE.value == "natural_language"
        assert QueryType.SEMANTIC.value == "semantic"
        assert QueryType.METRIC_RANGE.value == "metric_range"
        assert QueryType.TIME_SERIES.value == "time_series"


class TestKBDataType:
    """Test cases for KBDataType enum."""

    def test_data_type_values(self):
        """Test KBDataType enum values."""
        assert KBDataType.TELEMETRY_EVENT == "telemetry_event"
        assert KBDataType.METRIC_BATCH == "metric_batch"
        assert KBDataType.SYSTEM_STATE == "system_state"
        assert KBDataType.ALERT == "alert"
        assert KBDataType.CONFIGURATION == "configuration"
        assert KBDataType.GENERIC == "generic"

    def test_data_type_membership(self):
        """Test KBDataType membership checks."""
        assert KBDataType.TELEMETRY_EVENT in KBDataType
        assert KBDataType.GENERIC in KBDataType


class TestKBQuery:
    """Test cases for enhanced KBQuery model."""

    def test_kb_query_creation_minimal(self):
        """Test creating KBQuery with minimal required fields."""
        query = KBQuery(query_type=QueryType.STRUCTURED, content="test query")

        assert query.query_type == QueryType.STRUCTURED
        assert query.content == "test query"
        assert query.query_id is not None
        assert query.timestamp is not None
        assert query.filters is None
        assert query.metric_name is None
        assert query.time_range is None
        assert query.sources is None
        assert query.tags is None
        assert query.limit == 100
        assert query.offset == 0

    def test_kb_query_creation_telemetry_focused(self):
        """Test creating KBQuery with telemetry-specific fields."""
        time_range = {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
        }
        sources = ["system-monitor", "app-metrics"]
        tags = {"environment": "production", "service": "api"}

        query = KBQuery(
            query_type=QueryType.METRIC_RANGE,
            content="",
            metric_name="cpu.usage",
            time_range=time_range,
            sources=sources,
            tags=tags,
            data_types=[KBDataType.TELEMETRY_EVENT],
            limit=50,
        )

        assert query.query_type == QueryType.METRIC_RANGE
        assert query.metric_name == "cpu.usage"
        assert query.time_range == time_range
        assert query.sources == sources
        assert query.tags == tags
        assert query.data_types == [KBDataType.TELEMETRY_EVENT]
        assert query.limit == 50

    def test_kb_query_validation_empty(self):
        """Test that query validation requires at least one parameter."""
        with pytest.raises(ValueError, match="Query must have at least one of"):
            KBQuery(query_type=QueryType.STRUCTURED)

    def test_kb_query_validation_with_content(self):
        """Test query validation passes with content."""
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE, content="find cpu metrics"
        )
        assert query.content == "find cpu metrics"

    def test_kb_query_validation_with_filters(self):
        """Test query validation passes with filters."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED, filters={"metric_name": "memory.usage"}
        )
        assert query.filters == {"metric_name": "memory.usage"}

    def test_kb_query_validation_with_metric_name(self):
        """Test query validation passes with metric_name."""
        query = KBQuery(query_type=QueryType.METRIC_RANGE, metric_name="disk.usage")
        assert query.metric_name == "disk.usage"

    def test_kb_query_validation_with_time_range(self):
        """Test query validation passes with time_range."""
        query = KBQuery(
            query_type=QueryType.TIME_SERIES,
            time_range={"start_time": "2024-01-01T00:00:00Z"},
        )
        assert query.time_range["start_time"] == "2024-01-01T00:00:00Z"

    def test_kb_query_serialization(self):
        """Test KBQuery serialization to dict."""
        query = KBQuery(
            query_type=QueryType.TIME_SERIES,
            metric_name="network.bytes",
            sources=["router-01"],
            time_range={"start_time": "2024-01-01T00:00:00Z"},
            limit=25,
        )

        query_dict = query.model_dump()

        assert query_dict["query_type"] == "time_series"
        assert query_dict["metric_name"] == "network.bytes"
        assert query_dict["sources"] == ["router-01"]
        assert query_dict["time_range"]["start_time"] == "2024-01-01T00:00:00Z"
        assert query_dict["limit"] == 25
        assert "query_id" in query_dict
        assert "timestamp" in query_dict


class TestKBEntry:
    """Test cases for enhanced KBEntry model."""

    def test_kb_entry_creation_minimal(self):
        """Test creating KBEntry with minimal required fields."""
        content = {"title": "Test Entry", "body": "Test content"}
        entry = KBEntry(entry_id="entry-123", content=content)

        assert entry.entry_id == "entry-123"
        assert entry.content == content
        assert entry.data_type == KBDataType.GENERIC
        assert entry.metric_name is None
        assert entry.metric_value is None
        assert entry.source is None
        assert entry.tags is None
        assert entry.labels is None
        assert entry.metadata is None
        assert entry.event_timestamp is None
        assert entry.created_at is not None
        assert entry.updated_at is not None

    def test_kb_entry_telemetry_creation(self):
        """Test creating KBEntry for telemetry data."""
        content = {
            "name": "cpu.usage",
            "value": 85.5,
            "timestamp": "2024-01-01T12:00:00Z",
            "source": "monitor-01",
            "unit": "percent",
            "tags": {"host": "web-server-01", "region": "us-west"},
        }

        entry = KBEntry(
            entry_id="telemetry-456",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=content,
        )

        # Test auto-extraction of telemetry fields
        assert entry.metric_name == "cpu.usage"
        assert entry.metric_value == 85.5
        assert entry.source == "monitor-01"
        assert entry.event_timestamp == "2024-01-01T12:00:00Z"
        assert entry.labels == {"host": "web-server-01", "region": "us-west"}

    def test_kb_entry_metric_name_normalization(self):
        """Test metric name normalization."""
        content = {"name": "CPU_Usage_Percent", "value": 75}
        entry = KBEntry(
            entry_id="normalize-test",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=content,
        )

        assert entry.metric_name == "cpu.usage.percent"

    def test_kb_entry_manual_telemetry_fields(self):
        """Test creating KBEntry with manually set telemetry fields."""
        content = {"data": "some content"}
        entry = KBEntry(
            entry_id="manual-telemetry",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=content,
            metric_name="custom.metric",
            metric_value=42.0,
            source="manual-source",
            event_timestamp="2024-01-01T15:30:00Z",
            labels={"env": "test"},
        )

        # Manual values should not be overridden
        assert entry.metric_name == "custom.metric"
        assert entry.metric_value == 42.0
        assert entry.source == "manual-source"
        assert entry.event_timestamp == "2024-01-01T15:30:00Z"
        assert entry.labels == {"env": "test"}

    def test_kb_entry_to_telemetry_event(self):
        """Test conversion back to telemetry event format."""
        content = {
            "name": "memory.usage",
            "value": 1024,
            "timestamp": "2024-01-01T10:00:00Z",
            "source": "system-monitor",
            "unit": "MB",
            "metadata": {"process": "webapp"},
        }

        entry = KBEntry(
            entry_id="telemetry-convert",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=content,
            labels={"host": "server-01"},
        )

        telemetry_event = entry.to_telemetry_event()

        assert telemetry_event is not None
        assert telemetry_event["name"] == "memory.usage"
        assert telemetry_event["value"] == 1024
        assert telemetry_event["timestamp"] == "2024-01-01T10:00:00Z"
        assert telemetry_event["source"] == "system-monitor"
        assert telemetry_event["unit"] == "MB"
        assert telemetry_event["tags"] == {"host": "server-01"}
        assert telemetry_event["metadata"] == {"process": "webapp"}

    def test_kb_entry_to_telemetry_event_non_telemetry(self):
        """Test that non-telemetry entries return None for telemetry conversion."""
        entry = KBEntry(
            entry_id="non-telemetry",
            data_type=KBDataType.GENERIC,
            content={"data": "test"},
        )

        telemetry_event = entry.to_telemetry_event()
        assert telemetry_event is None

    def test_kb_entry_serialization(self):
        """Test KBEntry serialization to dict."""
        content = {"name": "disk.usage", "value": 75.5}
        entry = KBEntry(
            entry_id="serialize-test",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=content,
            tags=["storage", "system"],
            labels={"mount": "/var/log"},
        )

        entry_dict = entry.model_dump()

        assert entry_dict["entry_id"] == "serialize-test"
        assert entry_dict["data_type"] == "telemetry_event"
        assert entry_dict["content"] == content
        assert entry_dict["metric_name"] == "disk.usage"
        assert entry_dict["metric_value"] == 75.5
        assert entry_dict["tags"] == ["storage", "system"]
        assert entry_dict["labels"] == {"mount": "/var/log"}


class TestKBResponse:
    """Test cases for enhanced KBResponse model."""

    def test_kb_response_creation_minimal(self):
        """Test creating KBResponse with minimal required fields."""
        response = KBResponse(query_id="query-123", success=True)

        assert response.query_id == "query-123"
        assert response.success is True
        assert response.results == []
        assert response.total_results == 0
        assert response.message is None
        assert response.metadata is None
        assert response.processing_time_ms is None
        assert response.metric_summary is None
        assert response.time_range_covered is None
        assert response.sources_found is None
        assert response.data_types_found is None

    def test_kb_response_with_telemetry_results(self):
        """Test KBResponse with telemetry results and auto-computed metadata."""
        # Create telemetry entries
        entries = [
            KBEntry(
                entry_id="tel-1",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 85.0,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "source": "monitor-01",
                },
            ),
            KBEntry(
                entry_id="tel-2",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 92.5,
                    "timestamp": "2024-01-01T10:05:00Z",
                    "source": "monitor-02",
                },
            ),
        ]

        response = KBResponse(
            query_id="telemetry-query",
            success=True,
            results=entries,
            total_results=2,
            processing_time_ms=15.5,
        )

        # Check auto-computed metadata
        assert response.sources_found == ["monitor-01", "monitor-02"]
        assert response.data_types_found == [KBDataType.TELEMETRY_EVENT]
        assert response.time_range_covered["start_time"] == "2024-01-01T10:00:00Z"
        assert response.time_range_covered["end_time"] == "2024-01-01T10:05:00Z"

        # Check metric summary
        assert response.metric_summary is not None
        assert response.metric_summary["count"] == 2
        assert response.metric_summary["min"] == 85.0
        assert response.metric_summary["max"] == 92.5
        assert response.metric_summary["avg"] == 88.75
        assert "cpu.usage" in response.metric_summary["unique_metrics"]

    def test_kb_response_mixed_data_types(self):
        """Test KBResponse with mixed data types."""
        entries = [
            KBEntry(
                entry_id="tel-1",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={"name": "memory.usage", "value": 1024},
                source="system-01",
            ),
            KBEntry(
                entry_id="alert-1",
                data_type=KBDataType.ALERT,
                content={"severity": "high", "message": "High CPU usage"},
                source="alert-system",
            ),
        ]

        response = KBResponse(
            query_id="mixed-query", success=True, results=entries, total_results=2
        )

        assert len(response.data_types_found) == 2
        assert KBDataType.TELEMETRY_EVENT in response.data_types_found
        assert KBDataType.ALERT in response.data_types_found
        assert len(response.sources_found) == 2

    def test_kb_response_error_case(self):
        """Test creating KBResponse for error cases."""
        response = KBResponse(
            query_id="error-query",
            success=False,
            message="Invalid metric name",
            metadata={"error_code": "INVALID_METRIC"},
            processing_time_ms=2.1,
        )

        assert response.query_id == "error-query"
        assert response.success is False
        assert response.results == []
        assert response.total_results == 0
        assert response.message == "Invalid metric name"
        assert response.metadata["error_code"] == "INVALID_METRIC"
        assert response.processing_time_ms == 2.1

    def test_kb_response_serialization(self):
        """Test KBResponse serialization."""
        entry = KBEntry(
            entry_id="serialize-entry",
            data_type=KBDataType.TELEMETRY_EVENT,
            content={"name": "network.latency", "value": 15.2},
            source="network-monitor",
        )

        response = KBResponse(
            query_id="serialize-test",
            success=True,
            results=[entry],
            total_results=1,
            message="Success",
            processing_time_ms=8.7,
        )

        response_dict = response.model_dump()

        assert response_dict["query_id"] == "serialize-test"
        assert response_dict["success"] is True
        assert len(response_dict["results"]) == 1
        assert response_dict["total_results"] == 1
        assert response_dict["message"] == "Success"
        assert response_dict["processing_time_ms"] == 8.7
        assert response_dict["sources_found"] == ["network-monitor"]


class TestTelemetryQueryBuilder:
    """Test cases for TelemetryQueryBuilder helper class."""

    def test_metric_range_query_basic(self):
        """Test basic metric range query builder."""
        query = TelemetryQueryBuilder.metric_range_query(
            metric_name="cpu.usage", min_value=70.0, max_value=95.0
        )

        assert query.query_type == QueryType.METRIC_RANGE
        assert query.metric_name == "cpu.usage"
        assert query.filters["metric_value__gte"] == 70.0
        assert query.filters["metric_value__lte"] == 95.0
        assert query.filters["data_type"] == KBDataType.TELEMETRY_EVENT

    def test_metric_range_query_with_time_and_sources(self):
        """Test metric range query with time range and sources."""
        time_range = {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
        }
        sources = ["monitor-01", "monitor-02"]

        query = TelemetryQueryBuilder.metric_range_query(
            metric_name="memory.usage",
            min_value=100.0,
            time_range=time_range,
            sources=sources,
        )

        assert query.metric_name == "memory.usage"
        assert query.time_range == time_range
        assert query.sources == sources
        assert query.filters["metric_value__gte"] == 100.0
        assert "metric_value__lte" not in query.filters

    def test_time_series_query(self):
        """Test time series query builder."""
        query = TelemetryQueryBuilder.time_series_query(
            metric_name="disk.io",
            start_time="2024-01-01T10:00:00Z",
            end_time="2024-01-01T11:00:00Z",
            sources=["storage-monitor"],
            tags={"disk": "sda1", "type": "read"},
        )

        assert query.query_type == QueryType.TIME_SERIES
        assert query.metric_name == "disk.io"
        assert query.time_range["start_time"] == "2024-01-01T10:00:00Z"
        assert query.time_range["end_time"] == "2024-01-01T11:00:00Z"
        assert query.sources == ["storage-monitor"]
        assert query.tags == {"disk": "sda1", "type": "read"}
        assert query.filters["data_type"] == KBDataType.TELEMETRY_EVENT

    def test_natural_language_query(self):
        """Test natural language query builder."""
        query = TelemetryQueryBuilder.natural_language_query(
            content="find high CPU usage events",
            data_types=[KBDataType.TELEMETRY_EVENT, KBDataType.ALERT],
        )

        assert query.query_type == QueryType.NATURAL_LANGUAGE
        assert query.content == "find high CPU usage events"
        assert query.data_types == [KBDataType.TELEMETRY_EVENT, KBDataType.ALERT]

    def test_natural_language_query_default_data_types(self):
        """Test natural language query with default data types."""
        query = TelemetryQueryBuilder.natural_language_query(
            content="show memory metrics"
        )

        assert query.data_types == [KBDataType.TELEMETRY_EVENT]


class TestModelIntegration:
    """Test integration between enhanced models."""

    def test_complete_telemetry_flow(self):
        """Test complete telemetry query-response flow."""
        # Build query using TelemetryQueryBuilder
        query = TelemetryQueryBuilder.metric_range_query(
            metric_name="cpu.usage",
            min_value=80.0,
            time_range={
                "start_time": "2024-01-01T10:00:00Z",
                "end_time": "2024-01-01T11:00:00Z",
            },
            sources=["system-monitor"],
        )

        # Create matching telemetry entries
        entries = [
            KBEntry(
                entry_id="cpu-high-1",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 85.5,
                    "timestamp": "2024-01-01T10:15:00Z",
                    "source": "system-monitor",
                },
                labels={"host": "web-01"},
            ),
            KBEntry(
                entry_id="cpu-high-2",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={
                    "name": "cpu.usage",
                    "value": 92.3,
                    "timestamp": "2024-01-01T10:30:00Z",
                    "source": "system-monitor",
                },
                labels={"host": "web-02"},
            ),
        ]

        # Create response
        response = KBResponse(
            query_id=query.query_id,
            success=True,
            results=entries,
            total_results=2,
            processing_time_ms=12.5,
        )

        # Verify the complete flow
        assert response.query_id == query.query_id
        assert len(response.results) == 2
        assert all(e.metric_name == "cpu.usage" for e in response.results)
        assert all(e.metric_value >= 80.0 for e in response.results)
        assert response.metric_summary["min"] == 85.5
        assert response.metric_summary["max"] == 92.3
        assert response.sources_found == ["system-monitor"]

    def test_telemetry_event_round_trip(self):
        """Test converting telemetry event through KBEntry and back."""
        original_event = {
            "name": "network.bytes.sent",
            "value": 1048576,
            "timestamp": "2024-01-01T12:00:00Z",
            "source": "network-monitor",
            "unit": "bytes",
            "tags": {"interface": "eth0", "direction": "out"},
            "metadata": {"protocol": "tcp"},
        }

        # Convert to KBEntry
        entry = KBEntry(
            entry_id="network-test",
            data_type=KBDataType.TELEMETRY_EVENT,
            content=original_event,
        )

        # Verify auto-extraction worked
        assert entry.metric_name == "network.bytes.sent"
        assert entry.metric_value == 1048576
        assert entry.source == "network-monitor"
        assert entry.event_timestamp == "2024-01-01T12:00:00Z"
        assert entry.labels == {"interface": "eth0", "direction": "out"}

        # Convert back to telemetry event
        converted_event = entry.to_telemetry_event()

        # Verify round-trip conversion
        assert converted_event["name"] == original_event["name"]
        assert converted_event["value"] == original_event["value"]
        assert converted_event["timestamp"] == original_event["timestamp"]
        assert converted_event["source"] == original_event["source"]
        assert converted_event["unit"] == original_event["unit"]
        assert converted_event["tags"] == original_event["tags"]
        assert converted_event["metadata"] == original_event["metadata"]


if __name__ == "__main__":
    pytest.main([__file__])
