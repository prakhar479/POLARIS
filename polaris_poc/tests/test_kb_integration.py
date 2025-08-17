"""
Integration tests for Enhanced Knowledge Base System.

Tests the integration between telemetry-focused query models, enhanced storage,
and end-to-end workflows in the POLARIS Knowledge Base system.
"""

import pytest
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import time
import uuid

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
from polaris.common.kb_storage import InMemoryKBStorage


class EnhancedKnowledgeBaseService:
    """Enhanced Knowledge Base service with telemetry support for integration testing."""

    def __init__(self, storage=None, logger=None):
        self.storage = storage or InMemoryKBStorage(logger)
        self.logger = logger or logging.getLogger(__name__)
        self._query_count = 0

    def process_query(self, query: KBQuery) -> KBResponse:
        """Process a knowledge base query using the enhanced storage system."""
        self._query_count += 1

        try:
            # Use the storage's native query method
            response = self.storage.query(query)

            # Add service-level metadata
            if response.metadata is None:
                response.metadata = {}

            response.metadata.update(
                {"query_number": self._query_count, "service_processed": True}
            )

            return response

        except Exception as e:
            self.logger.error(f"Service-level query processing failed: {e}")
            return KBResponse(
                query_id=query.query_id,
                success=False,
                message=f"Service error: {str(e)}",
                metadata={"query_number": self._query_count},
            )

    def add_entry(self, entry: KBEntry) -> bool:
        """Add entry to knowledge base."""
        return self.storage.store(entry)

    def add_telemetry_event(self, telemetry_event: dict) -> bool:
        """Add telemetry event directly."""
        return self.storage.store_telemetry_event(telemetry_event)

    def add_telemetry_batch(self, batch: dict) -> dict:
        """Add batch of telemetry events."""
        return self.storage.store_telemetry_batch(batch)

    def update_entry(self, entry_id: str, updates: dict) -> bool:
        """Update existing entry."""
        return self.storage.update(entry_id, updates)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete entry from knowledge base."""
        return self.storage.delete(entry_id)

    def get_statistics(self):
        """Get comprehensive system statistics."""
        storage_stats = self.storage.get_stats()
        return {**storage_stats, "total_queries_processed": self._query_count}

    def get_metric_summary(self, metric_name: str):
        """Get metric summary statistics."""
        return self.storage.get_metric_summary(metric_name)


class TestTelemetryKnowledgeBaseIntegration:
    """Test telemetry-focused knowledge base system integration."""

    def setup_method(self):
        """Set up test fixtures with telemetry data."""
        self.logger = Mock(spec=logging.Logger)
        self.kb_service = EnhancedKnowledgeBaseService(logger=self.logger)

        # Add sample telemetry and system data
        self._setup_sample_data()

    def _setup_sample_data(self):
        """Set up comprehensive sample data including telemetry."""

        # Telemetry events
        telemetry_events = [
            {
                "name": "cpu.usage",
                "value": 85.5,
                "timestamp": "2024-01-01T10:00:00Z",
                "source": "system-monitor",
                "unit": "percent",
                "tags": {"host": "web-01", "env": "prod"},
            },
            {
                "name": "cpu.usage",
                "value": 92.3,
                "timestamp": "2024-01-01T10:05:00Z",
                "source": "system-monitor",
                "unit": "percent",
                "tags": {"host": "web-02", "env": "prod"},
            },
            {
                "name": "memory.usage",
                "value": 1024,
                "timestamp": "2024-01-01T10:00:00Z",
                "source": "system-monitor",
                "unit": "MB",
                "tags": {"host": "web-01", "env": "prod"},
            },
            {
                "name": "disk.usage",
                "value": 75.2,
                "timestamp": "2024-01-01T10:00:00Z",
                "source": "storage-monitor",
                "unit": "percent",
                "tags": {"mount": "/var/log", "host": "db-01"},
            },
            {
                "name": "network.latency",
                "value": 15.8,
                "timestamp": "2024-01-01T10:00:00Z",
                "source": "network-monitor",
                "unit": "ms",
                "tags": {"interface": "eth0", "host": "web-01"},
            },
        ]

        for event in telemetry_events:
            self.kb_service.add_telemetry_event(event)

        # System alerts and other data types
        other_entries = [
            KBEntry(
                entry_id="alert-001",
                data_type=KBDataType.ALERT,
                content={
                    "title": "High CPU Alert",
                    "description": "CPU usage exceeded threshold",
                    "severity": "critical",
                    "threshold": 90.0,
                    "current_value": 92.3,
                },
                source="alert-system",
                tags=["alert", "cpu", "critical"],
                labels={"host": "web-02", "env": "prod"},
            ),
            KBEntry(
                entry_id="config-001",
                data_type=KBDataType.CONFIGURATION,
                content={
                    "title": "Load Balancer Config",
                    "description": "Updated load balancing configuration",
                    "config_type": "nginx",
                    "version": "1.2.3",
                },
                source="config-manager",
                tags=["config", "nginx", "load-balancer"],
                labels={"env": "prod", "service": "web"},
            ),
            KBEntry(
                entry_id="state-001",
                data_type=KBDataType.SYSTEM_STATE,
                content={
                    "title": "System Health Snapshot",
                    "description": "Periodic system health check",
                    "overall_status": "healthy",
                    "components": {
                        "web_servers": "healthy",
                        "database": "healthy",
                        "load_balancer": "healthy",
                    },
                },
                source="health-monitor",
                tags=["health", "system", "monitoring"],
                labels={"env": "prod"},
            ),
        ]

        for entry in other_entries:
            self.kb_service.add_entry(entry)

    def test_metric_range_query_integration(self):
        """Test metric range query end-to-end."""
        query = TelemetryQueryBuilder.metric_range_query(
            metric_name="cpu.usage",
            min_value=80.0,
            max_value=95.0,
            sources=["system-monitor"],
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.query_id == query.query_id
        assert response.total_results == 2
        assert len(response.results) == 2

        # Verify all results are CPU metrics within range
        for result in response.results:
            assert result.metric_name == "cpu.usage"
            assert 80.0 <= result.metric_value <= 95.0
            assert result.source == "system-monitor"

        # Check response metadata
        assert response.metric_summary is not None
        assert response.metric_summary["min"] == 85.5
        assert response.metric_summary["max"] == 92.3
        assert response.sources_found == ["system-monitor"]
        assert response.processing_time_ms > 0

    def test_time_series_query_integration(self):
        """Test time series query processing."""
        query = TelemetryQueryBuilder.time_series_query(
            metric_name="memory.usage",
            start_time="2024-01-01T09:59:00Z",
            end_time="2024-01-01T10:01:00Z",
            sources=["system-monitor"],
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].metric_name == "memory.usage"
        assert response.results[0].metric_value == 1024
        assert response.time_range_covered is not None
        assert response.time_range_covered["start_time"] == "2024-01-01T10:00:00Z"

    def test_natural_language_telemetry_query(self):
        """Test natural language query on telemetry data."""
        query = TelemetryQueryBuilder.natural_language_query(
            content="disk usage", data_types=[KBDataType.TELEMETRY_EVENT]
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].metric_name == "disk.usage"
        assert response.results[0].source == "storage-monitor"

    def test_mixed_data_type_query(self):
        """Test querying across multiple data types."""
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            content="cpu",
            data_types=[KBDataType.TELEMETRY_EVENT, KBDataType.ALERT],
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results >= 2  # At least telemetry events and alert

        # Should include both telemetry events and alerts
        data_types_found = {result.data_type for result in response.results}
        assert KBDataType.TELEMETRY_EVENT in data_types_found
        assert KBDataType.ALERT in data_types_found

    def test_structured_query_with_labels(self):
        """Test structured query with label filtering."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"data_type": KBDataType.TELEMETRY_EVENT, "labels.host": "web-01"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results >= 2  # CPU and memory from web-01

        # Verify all results are from web-01
        for result in response.results:
            assert result.labels.get("host") == "web-01"

    def test_complex_telemetry_filters(self):
        """Test complex telemetry filtering with operators."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={
                "data_type": KBDataType.TELEMETRY_EVENT,
                "metric_value__gte": 80.0,  # Value >= 80
                "labels.env": "prod",  # Production environment
            },
        )

        response = self.kb_service.process_query(query)

        assert response.success is True

        # Verify all results meet criteria
        for result in response.results:
            assert result.metric_value >= 80.0
            assert result.labels.get("env") == "prod"

    def test_telemetry_batch_processing(self):
        """Test batch telemetry event processing."""
        batch = {
            "events": [
                {
                    "name": "batch.metric.1",
                    "value": 10.5,
                    "timestamp": "2024-01-01T11:00:00Z",
                    "source": "batch-test",
                },
                {
                    "name": "batch.metric.2",
                    "value": 20.3,
                    "timestamp": "2024-01-01T11:00:00Z",
                    "source": "batch-test",
                },
            ]
        }

        results = self.kb_service.add_telemetry_batch(batch)
        assert results["stored"] == 2
        assert results["failed"] == 0

        # Query the batch metrics
        query = KBQuery(
            query_type=QueryType.STRUCTURED, filters={"source": "batch-test"}
        )

        response = self.kb_service.process_query(query)
        assert response.success is True
        assert response.total_results == 2

    def test_metric_summary_integration(self):
        """Test metric summary functionality."""
        summary = self.kb_service.get_metric_summary("cpu.usage")

        assert summary["metric_name"] == "cpu.usage"
        assert summary["total_entries"] == 2
        assert summary["value_statistics"]["count"] == 2
        assert summary["value_statistics"]["min"] == 85.5
        assert summary["value_statistics"]["max"] == 92.3
        assert summary["value_statistics"]["avg"] == (85.5 + 92.3) / 2
        assert "system-monitor" in summary["unique_sources"]

    def test_pagination_integration(self):
        """Test query pagination."""
        # Query with limit
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"data_type": KBDataType.TELEMETRY_EVENT},
            limit=2,
            offset=0,
        )

        response = self.kb_service.process_query(query)
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 5  # Total telemetry events

        # Query next page
        query.offset = 2
        response = self.kb_service.process_query(query)
        assert len(response.results) == 2
        assert response.total_results == 5


class TestTelemetryKnowledgeBaseCRUD:
    """Test CRUD operations with telemetry data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kb_service = EnhancedKnowledgeBaseService()

    def test_telemetry_entry_lifecycle(self):
        """Test complete telemetry entry lifecycle."""
        # Add telemetry event
        event = {
            "name": "lifecycle.test",
            "value": 42.0,
            "timestamp": "2024-01-01T12:00:00Z",
            "source": "test-source",
            "tags": {"test": "lifecycle"},
        }

        assert self.kb_service.add_telemetry_event(event) is True

        # Query for the event
        query = TelemetryQueryBuilder.metric_range_query(metric_name="lifecycle.test")

        response = self.kb_service.process_query(query)
        assert response.success is True
        assert response.total_results == 1

        entry = response.results[0]
        assert entry.metric_name == "lifecycle.test"
        assert entry.metric_value == 42.0

        # Update the entry
        updates = {
            "content": {"updated": True},
            "labels": {"test": "lifecycle", "status": "updated"},
        }
        assert self.kb_service.update_entry(entry.entry_id, updates) is True

        # Verify update
        response = self.kb_service.process_query(query)
        updated_entry = response.results[0]
        assert updated_entry.content.get("updated") is True
        assert updated_entry.labels.get("status") == "updated"

        # Delete entry
        assert self.kb_service.delete_entry(entry.entry_id) is True

        # Verify deletion
        response = self.kb_service.process_query(query)
        assert response.total_results == 0

    def test_mixed_data_types_crud(self):
        """Test CRUD operations across different data types."""
        entries = [
            KBEntry(
                entry_id="tel-crud",
                data_type=KBDataType.TELEMETRY_EVENT,
                content={"name": "crud.test", "value": 100},
                metric_name="crud.test",
                metric_value=100,
            ),
            KBEntry(
                entry_id="alert-crud",
                data_type=KBDataType.ALERT,
                content={"title": "CRUD Test Alert", "severity": "info"},
            ),
            KBEntry(
                entry_id="config-crud",
                data_type=KBDataType.CONFIGURATION,
                content={"name": "CRUD Config", "version": "1.0"},
            ),
        ]

        # Add all entries
        for entry in entries:
            assert self.kb_service.add_entry(entry) is True

        # Query by data type
        for data_type in [
            KBDataType.TELEMETRY_EVENT,
            KBDataType.ALERT,
            KBDataType.CONFIGURATION,
        ]:
            query = KBQuery(
                query_type=QueryType.STRUCTURED, filters={"data_type": data_type}
            )
            response = self.kb_service.process_query(query)
            assert response.success is True
            assert response.total_results == 1
            assert response.results[0].data_type == data_type

        # Update all entries
        for entry in entries:
            updates = {"metadata": {"batch_updated": True}}
            assert self.kb_service.update_entry(entry.entry_id, updates) is True

        # Delete all entries
        for entry in entries:
            assert self.kb_service.delete_entry(entry.entry_id) is True


class TestTelemetryKnowledgeBaseErrorHandling:
    """Test error handling in telemetry-focused system."""

    def setup_method(self):
        """Set up test fixtures with mock storage."""
        self.logger = Mock(spec=logging.Logger)
        self.failing_storage = Mock(spec=InMemoryKBStorage)
        self.kb_service = EnhancedKnowledgeBaseService(
            storage=self.failing_storage, logger=self.logger
        )

    def test_telemetry_storage_failure(self):
        """Test handling of telemetry storage failures."""
        # Mock storage to raise exception
        self.failing_storage.query.side_effect = Exception("Telemetry storage failed")

        query = TelemetryQueryBuilder.metric_range_query(metric_name="test.metric")

        response = self.kb_service.process_query(query)

        assert response.success is False
        assert "Service error" in response.message
        assert "Telemetry storage failed" in response.message
        assert response.metadata["query_number"] == 1

        # Verify error was logged
        self.logger.error.assert_called_once()

    def test_invalid_telemetry_data_handling(self):
        """Test handling of invalid telemetry data."""
        # Mock storage methods for adding data
        self.failing_storage.store_telemetry_event.return_value = False

        # Try to add invalid telemetry event
        invalid_event = {"invalid": "data"}
        result = self.kb_service.add_telemetry_event(invalid_event)
        assert result is False

    def test_query_validation_errors(self):
        """Test handling of query validation errors."""
        # This should be handled at the query model level,
        # but test service resilience
        try:
            invalid_query = KBQuery(
                query_type=QueryType.METRIC_RANGE
                # Missing required fields - should fail validation
            )
            assert False, "Should have failed validation"
        except ValueError:
            # Expected validation error
            pass


class TestTelemetryKnowledgeBasePerformance:
    """Test performance with telemetry-focused workloads."""

    def setup_method(self):
        """Set up test fixtures with large telemetry dataset."""
        self.kb_service = EnhancedKnowledgeBaseService()

        # Create realistic telemetry dataset
        self._create_large_telemetry_dataset()

    def _create_large_telemetry_dataset(self):
        """Create a large dataset of telemetry events."""
        import random
        from datetime import timedelta

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        metrics = [
            "cpu.usage",
            "memory.usage",
            "disk.usage",
            "network.latency",
            "disk.io",
        ]
        sources = ["monitor-01", "monitor-02", "monitor-03"]
        hosts = ["web-01", "web-02", "db-01", "cache-01"]

        # Add 500 telemetry events
        for i in range(500):
            event_time = base_time + timedelta(minutes=i // 10)

            event = {
                "name": random.choice(metrics),
                "value": random.uniform(10, 100),
                "timestamp": event_time.isoformat(),
                "source": random.choice(sources),
                "tags": {"host": random.choice(hosts), "env": "test", "index": str(i)},
            }
            self.kb_service.add_telemetry_event(event)

        # Add some non-telemetry data
        for i in range(50):
            entry = KBEntry(
                entry_id=f"perf-entry-{i}",
                data_type=random.choice(
                    [
                        KBDataType.ALERT,
                        KBDataType.CONFIGURATION,
                        KBDataType.SYSTEM_STATE,
                    ]
                ),
                content={
                    "title": f"Performance Test Entry {i}",
                    "index": i,
                    "category": f"cat{i % 5}",
                },
                tags=[f"tag{i % 3}", "performance"],
            )
            self.kb_service.add_entry(entry)

    def test_telemetry_query_performance(self):
        """Test performance of various telemetry queries."""
        queries = [
            # Metric range query
            TelemetryQueryBuilder.metric_range_query(
                metric_name="cpu.usage", min_value=50.0
            ),
            # Time series query
            TelemetryQueryBuilder.time_series_query(
                metric_name="memory.usage",
                start_time="2024-01-01T10:00:00Z",
                end_time="2024-01-01T12:00:00Z",
            ),
            # Natural language query
            TelemetryQueryBuilder.natural_language_query(content="network latency"),
            # Complex structured query
            KBQuery(
                query_type=QueryType.STRUCTURED,
                filters={
                    "data_type": KBDataType.TELEMETRY_EVENT,
                    "metric_value__gte": 75.0,
                    "labels.env": "test",
                },
            ),
        ]

        processing_times = []
        for query in queries:
            response = self.kb_service.process_query(query)
            assert response.success is True
            processing_times.append(response.processing_time_ms)

        # Verify reasonable processing times (under 50ms for this dataset)
        for time_ms in processing_times:
            assert time_ms < 50, f"Query took too long: {time_ms}ms"

        # Verify comprehensive statistics
        stats = self.kb_service.get_statistics()
        assert stats["total_entries"] == 550  # 500 telemetry + 50 others
        assert stats["data_type_breakdown"]["telemetry_event"] == 500

    def test_concurrent_telemetry_queries(self):
        """Test simulated concurrent telemetry query processing."""
        # Create diverse telemetry queries
        queries = []
        metrics = ["cpu.usage", "memory.usage", "disk.usage", "network.latency"]

        for i in range(50):
            metric = metrics[i % len(metrics)]
            if i % 3 == 0:
                # Metric range query
                query = TelemetryQueryBuilder.metric_range_query(
                    metric_name=metric, min_value=float(i % 50)
                )
            elif i % 3 == 1:
                # Time series query
                start_time = f"2024-01-01T{10 + i % 12:02d}:00:00Z"
                end_time = f"2024-01-01T{10 + (i % 12) + 1:02d}:00:00Z"
                query = TelemetryQueryBuilder.time_series_query(
                    metric_name=metric, start_time=start_time, end_time=end_time
                )
            else:
                # Natural language query
                query = TelemetryQueryBuilder.natural_language_query(
                    content=metric.replace(".", " ")
                )
            queries.append(query)

        # Process all queries and measure time
        responses = []
        start_time = time.time()

        for query in queries:
            response = self.kb_service.process_query(query)
            responses.append(response)

        total_time = time.time() - start_time

        # Verify all succeeded
        assert all(r.success for r in responses)

        # Verify reasonable total processing time
        assert total_time < 5.0, f"Total processing took too long: {total_time}s"

        # Verify statistics
        stats = self.kb_service.get_statistics()
        assert stats["total_queries_processed"] == len(queries)

    def test_telemetry_metric_summary_performance(self):
        """Test performance of metric summary operations."""
        # Test summaries for different metrics
        metrics = [
            "cpu.usage",
            "memory.usage",
            "disk.usage",
            "network.latency",
            "disk.io",
        ]

        summaries = []
        start_time = time.time()

        for metric in metrics:
            summary = self.kb_service.get_metric_summary(metric)
            summaries.append(summary)

        total_time = time.time() - start_time

        # Verify reasonable processing time
        assert total_time < 1.0, f"Summary generation took too long: {total_time}s"

        # Verify summaries contain expected data
        for summary in summaries:
            if "error" not in summary:  # Some metrics might not exist
                assert "value_statistics" in summary
                assert "total_entries" in summary
                assert summary["total_entries"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
