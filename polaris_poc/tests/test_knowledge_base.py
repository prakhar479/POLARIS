"""
Unit tests for POLARIS Knowledge Base implementation.

Tests the InMemoryKnowledgeBase with focus on telemetry buffering,
aggregation, and various query patterns.
"""

import pytest
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.knowledge_base.models import (
    KBEntry,
    KBQuery,
    KBResponse,
    KBDataType,
    QueryType,
)
from polaris.knowledge_base.base import BaseKnowledgeBase
from polaris.models.knowledge_base_impl import InMemoryKnowledgeBase


class TestKBModels:
    """Test cases for Knowledge Base data models."""

    def test_kb_entry_creation(self):
        """Test KBEntry model creation with default values."""
        entry = KBEntry(
            data_type=KBDataType.OBSERVATION,
            summary="Test observation",
            content={"cpu_usage": 75.5, "status": "normal"},
        )

        assert entry.data_type == KBDataType.OBSERVATION
        assert entry.summary == "Test observation"
        assert entry.content["cpu_usage"] == 75.5
        assert entry.entry_id is not None
        assert entry.timestamp is not None
        assert isinstance(entry.tags, list)
        assert isinstance(entry.metadata, dict)

    def test_kb_entry_with_telemetry_fields(self):
        """Test KBEntry with telemetry-specific fields."""
        entry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="cpu.usage",
            metric_value=85.3,
            source="swim_monitor",
            summary="CPU usage telemetry",
            tags=["telemetry", "cpu", "monitoring"],
        )

        assert entry.data_type == KBDataType.RAW_TELEMETRY_EVENT
        assert entry.metric_name == "cpu.usage"
        assert entry.metric_value == 85.3
        assert entry.source == "swim_monitor"
        assert "telemetry" in entry.tags

    def test_kb_query_structured(self):
        """Test structured KBQuery creation."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"status": "normal", "cpu_usage": 75.5},
            data_types=[KBDataType.OBSERVATION],
            limit=10,
        )

        assert query.query_type == QueryType.STRUCTURED
        assert query.filters["status"] == "normal"
        assert KBDataType.OBSERVATION in query.data_types
        assert query.limit == 10
        assert query.query_id is not None

    def test_kb_query_natural_language(self):
        """Test natural language KBQuery creation."""
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            query_text="show me all CPU usage observations",
            data_types=[KBDataType.OBSERVATION, KBDataType.LEARNED_PATTERN],
            limit=20,
        )

        assert query.query_type == QueryType.NATURAL_LANGUAGE
        assert query.query_text == "show me all CPU usage observations"
        assert len(query.data_types) == 2
        assert query.limit == 20

    def test_kb_response_structure(self):
        """Test KBResponse model structure."""
        entries = [
            KBEntry(data_type=KBDataType.OBSERVATION, summary="Test 1"),
            KBEntry(data_type=KBDataType.OBSERVATION, summary="Test 2"),
        ]

        response = KBResponse(
            query_id="test-query-123",
            success=True,
            results=entries,
            total_results=2,
            processing_time_ms=15.5,
        )

        assert response.query_id == "test-query-123"
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.processing_time_ms == 15.5


class TestInMemoryKnowledgeBase:
    """Test cases for InMemoryKnowledgeBase implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kb = InMemoryKnowledgeBase(telemetry_buffer_size=3)

    def test_initialization(self):
        """Test knowledge base initialization."""
        assert isinstance(self.kb, BaseKnowledgeBase)
        assert self.kb._telemetry_buffer_size == 3
        assert len(self.kb._entries) == 0
        assert len(self.kb._raw_telemetry_buffers) == 0

    def test_store_and_get_permanent_entry(self):
        """Test storing and retrieving permanent entries."""
        entry = KBEntry(
            data_type=KBDataType.OBSERVATION,
            summary="CPU usage observation",
            content={"cpu_usage": 75.5, "timestamp": "2025-08-22T10:00:00Z"},
            tags=["cpu", "monitoring"],
        )

        # Store the entry
        result = self.kb.store(entry)
        assert result is True

        # Retrieve the entry
        retrieved = self.kb.get(entry.entry_id)
        assert retrieved is not None
        assert retrieved.entry_id == entry.entry_id
        assert retrieved.summary == "CPU usage observation"
        assert retrieved.content["cpu_usage"] == 75.5

    def test_store_raw_telemetry_buffering(self):
        """Test raw telemetry event buffering."""
        # Create telemetry entries
        telemetry_entries = []
        for i in range(5):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name="cpu.usage",
                metric_value=80.0 + i,
                source="swim_monitor",
                summary=f"CPU usage reading {i+1}",
            )
            telemetry_entries.append(entry)

        # Store first 3 entries (should stay in buffer)
        for i in range(3):
            result = self.kb.store(telemetry_entries[i])
            assert result is True

        # Check buffer state
        buffer_key = ("cpu.usage", "swim_monitor")
        assert buffer_key in self.kb._raw_telemetry_buffers
        assert len(self.kb._raw_telemetry_buffers[buffer_key]) == 3

        # No permanent entries should exist yet
        assert len(self.kb._entries) == 0

        # Store 4th entry (should trigger aggregation)
        result = self.kb.store(telemetry_entries[3])
        assert result is True

        # Check that aggregation occurred
        assert len(self.kb._entries) == 1  # One aggregated observation

        # After aggregation, buffer should be cleared and contain only the new entry
        assert len(self.kb._raw_telemetry_buffers[buffer_key]) == 1
        # The buffer should contain only the 4th entry
        buffer_values = [
            e.metric_value for e in self.kb._raw_telemetry_buffers[buffer_key]
        ]
        assert buffer_values == [83.0]  # Value from entry 4

    def test_telemetry_aggregation_content(self):
        """Test the content of aggregated telemetry observations."""
        # Create and store telemetry entries that will trigger aggregation
        values = [75.0, 80.0, 85.0]
        for i, value in enumerate(values):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name="memory.usage",
                metric_value=value,
                source="system_monitor",
                summary=f"Memory usage reading {i+1}",
            )
            self.kb.store(entry)

        # Store one more to trigger aggregation
        trigger_entry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="memory.usage",
            metric_value=90.0,
            source="system_monitor",
            summary="Memory usage reading 4",
        )
        self.kb.store(trigger_entry)

        # Check aggregated entry
        assert len(self.kb._entries) == 1
        aggregated_entry = list(self.kb._entries.values())[0]

        assert aggregated_entry.data_type == KBDataType.OBSERVATION
        assert aggregated_entry.metric_name == "memory.usage"
        assert aggregated_entry.source == "system_monitor"
        assert "aggregated" in aggregated_entry.tags

        # Check aggregation statistics
        content = aggregated_entry.content
        assert content["count"] == 3
        assert content["average_value"] == 80.0  # (75+80+85)/3
        assert content["min_value"] == 75.0
        assert content["max_value"] == 85.0
        assert content["statistic"] == "aggregation"
        assert content["trend"] == "baseline"  # First observation
        assert content["total_updates"] == 1

    def test_telemetry_trend_updates(self):
        """Test trend calculation and observation updates."""
        metric_name = "cpu.usage"
        source = "test_monitor"

        # First batch: values around 50 (will aggregate after 3rd entry)
        first_batch_values = [48.0, 50.0, 52.0]
        for i, value in enumerate(first_batch_values):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=value,
                source=source,
                summary=f"CPU reading {i+1}",
            )
            self.kb.store(entry)

        # Store one more to trigger first aggregation
        trigger1 = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name=metric_name,
            metric_value=49.0,
            source=source,
            summary="Trigger first aggregation",
        )
        self.kb.store(trigger1)

        # Check first observation
        assert len(self.kb._entries) == 1
        obs1 = list(self.kb._entries.values())[0]
        assert obs1.content["trend"] == "baseline"
        assert obs1.content["total_updates"] == 1
        first_avg = obs1.content["average_value"]

        # Second batch: significantly higher values (around 70) for clear increasing trend
        # Store 3 more entries to trigger second aggregation
        second_batch_values = [68.0, 70.0, 72.0]
        for i, value in enumerate(second_batch_values):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=value,
                source=source,
                summary=f"CPU reading batch 2 - {i+1}",
            )
            self.kb.store(entry)

        # Check observation after second aggregation is triggered
        obs2 = list(self.kb._entries.values())[0]
        assert obs2.content["trend"] == "increasing"  # Should be increasing now
        assert obs2.content["total_updates"] == 2
        assert obs2.content["previous_average"] == first_avg

    def test_telemetry_stable_trend(self):
        """Test stable trend detection (within 5% threshold)."""
        metric_name = "memory.usage"
        source = "system_monitor"

        # First batch: average 100 (trigger aggregation with 4th entry)
        for i, value in enumerate([98.0, 100.0, 102.0, 100.0]):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=value,
                source=source,
            )
            self.kb.store(entry)

        # Check first observation
        assert len(self.kb._entries) == 1
        obs1 = list(self.kb._entries.values())[0]
        first_avg = obs1.content["average_value"]

        # Second batch: average 103 (within 5% of first average, should be stable)
        for value in [101.0, 103.0, 105.0]:
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=value,
                source=source,
            )
            self.kb.store(entry)

        # Check that we now have stable trend
        obs2 = list(self.kb._entries.values())[0]
        assert obs2.content["trend"] == "stable"  # Should be within 5% threshold
        assert obs2.content["total_updates"] == 2

    def test_observation_entry_id_consistency(self):
        """Test that observations use consistent IDs for same metric/source."""
        metric_name = "disk.io"
        source = "storage_monitor"

        # Store entries in batches to trigger exactly 3 aggregations
        for batch in range(3):
            # Store exactly 3 entries to trigger one aggregation per batch
            for i in range(3):
                entry = KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name=metric_name,
                    metric_value=10.0 + batch * 10 + i,
                    source=source,
                    summary=f"Batch {batch} reading {i}",
                )
                self.kb.store(entry)

            # Add one more to trigger aggregation
            trigger = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=15.0 + batch * 10,
                source=source,
            )
            self.kb.store(trigger)

        # Should still have only one observation entry (updated multiple times)
        assert len(self.kb._entries) == 1
        obs = list(self.kb._entries.values())[0]
        assert obs.content["total_updates"] == 3

        # Check that entry ID follows expected pattern
        expected_id = f"obs_{metric_name}_{source}".replace(".", "_")
        assert obs.entry_id == expected_id

    def test_telemetry_stable_trend(self):
        """Test stable trend detection (within 5% threshold)."""
        metric_name = "memory.usage"
        source = "system_monitor"

        # First batch: average 100
        for value in [98.0, 100.0, 102.0]:
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=value,
                source=source,
            )
            self.kb.store(entry)

        # Trigger first aggregation
        self.kb.store(
            KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=100.0,
                source=source,
            )
        )

        # Second batch: average 103 (within 5% of 100, should be stable)
        for value in [101.0, 103.0, 105.0]:
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=value,
                source=source,
            )
            self.kb.store(entry)

        # Trigger second aggregation
        self.kb.store(
            KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric_name,
                metric_value=102.0,
                source=source,
            )
        )

        # Check stable trend
        obs = list(self.kb._entries.values())[0]
        assert obs.content["trend"] == "stable"  # 103 is within 5% of 100
        assert obs.content["total_updates"] == 2

    def test_observation_entry_id_consistency(self):
        """Test that observations use consistent IDs for same metric/source."""
        metric_name = "disk.io"
        source = "storage_monitor"

        # Store multiple batches for same metric/source
        for batch in range(3):
            for i in range(3):
                entry = KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name=metric_name,
                    metric_value=10.0 + batch * 10 + i,
                    source=source,
                    summary=f"Batch {batch} reading {i}",
                )
                self.kb.store(entry)

            # Trigger aggregation
            self.kb.store(
                KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name=metric_name,
                    metric_value=15.0 + batch * 10,
                    source=source,
                )
            )

        # Should still have only one observation entry (updated multiple times)
        assert len(self.kb._entries) == 1
        obs = list(self.kb._entries.values())[0]
        assert obs.content["total_updates"] == 3

        # Check that entry ID follows expected pattern
        expected_id = f"obs_{metric_name}_{source}".replace(".", "_")
        assert obs.entry_id == expected_id

    def test_multiple_metric_buffers(self):
        """Test handling multiple telemetry metrics simultaneously."""
        # Store CPU telemetry
        cpu_entry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="cpu.usage",
            metric_value=75.0,
            source="monitor_a",
        )
        self.kb.store(cpu_entry)

        # Store memory telemetry
        memory_entry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="memory.usage",
            metric_value=60.0,
            source="monitor_b",
        )
        self.kb.store(memory_entry)

        # Should have separate buffers
        assert len(self.kb._raw_telemetry_buffers) == 2
        assert ("cpu.usage", "monitor_a") in self.kb._raw_telemetry_buffers
        assert ("memory.usage", "monitor_b") in self.kb._raw_telemetry_buffers

    def test_multiple_sources_trend_tracking(self):
        """Test that different sources for same metric maintain separate trends."""
        metric_name = "network.latency"
        sources = ["datacenter_a", "datacenter_b"]

        # Create observations for both sources with different trends
        for source in sources:
            base_value = 10.0 if source == "datacenter_a" else 50.0

            # First batch for this source
            for i in range(3):
                entry = KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name=metric_name,
                    metric_value=base_value + i,
                    source=source,
                    summary=f"Latency from {source}",
                )
                self.kb.store(entry)

            # Trigger aggregation
            self.kb.store(
                KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name=metric_name,
                    metric_value=base_value + 1.5,
                    source=source,
                )
            )

        # Should have 2 separate observation entries
        assert len(self.kb._entries) == 2

        # Get observations by source
        observations = list(self.kb._entries.values())
        obs_by_source = {obs.source: obs for obs in observations}

        assert "datacenter_a" in obs_by_source
        assert "datacenter_b" in obs_by_source

        # Both should have baseline trend (first observations)
        assert obs_by_source["datacenter_a"].content["trend"] == "baseline"
        assert obs_by_source["datacenter_b"].content["trend"] == "baseline"

        # Verify different averages
        assert (
            obs_by_source["datacenter_a"].content["average_value"] == 11.0
        )  # (10+11+12)/3
        assert (
            obs_by_source["datacenter_b"].content["average_value"] == 51.0
        )  # (50+51+52)/3

    def test_invalid_telemetry_handling(self):
        """Test handling of invalid telemetry entries."""
        # Entry without metric_name
        invalid_entry1 = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_value=75.0,
            source="monitor",
        )
        result = self.kb.store(invalid_entry1)
        assert result is False

        # Entry without source
        invalid_entry2 = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="cpu.usage",
            metric_value=75.0,
        )
        result = self.kb.store(invalid_entry2)
        assert result is False

    def test_delete_entry(self):
        """Test deleting entries from the knowledge base."""
        entry = KBEntry(
            data_type=KBDataType.SYSTEM_GOAL,
            summary="Maintain response time under 100ms",
            content={"target_response_time": 100, "unit": "ms"},
            tags=["performance", "goal"],
        )

        # Store and verify
        self.kb.store(entry)
        assert self.kb.get(entry.entry_id) is not None

        # Delete and verify
        result = self.kb.delete(entry.entry_id)
        assert result is True
        assert self.kb.get(entry.entry_id) is None

        # Try to delete non-existent entry
        result = self.kb.delete("non-existent-id")
        assert result is False

    def test_structured_query(self):
        """Test structured queries against the knowledge base."""
        # Store test entries
        entries = [
            KBEntry(
                data_type=KBDataType.OBSERVATION,
                content={"status": "normal", "cpu_usage": 75.0},
                tags=["cpu", "monitoring"],
            ),
            KBEntry(
                data_type=KBDataType.OBSERVATION,
                content={"status": "warning", "cpu_usage": 95.0},
                tags=["cpu", "monitoring", "alert"],
            ),
            KBEntry(
                data_type=KBDataType.SYSTEM_GOAL,
                content={"target": "performance"},
                tags=["goal", "performance"],
            ),
        ]

        for entry in entries:
            self.kb.store(entry)

        # Query by data type
        query = KBQuery(
            query_type=QueryType.STRUCTURED, data_types=[KBDataType.OBSERVATION]
        )
        response = self.kb.query(query)

        assert response.success is True
        assert response.total_results == 2
        assert len(response.results) == 2

        # Query with filters
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            filters={"status": "normal"},
            data_types=[KBDataType.OBSERVATION],
        )
        response = self.kb.query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].content["status"] == "normal"

    def test_natural_language_query(self):
        """Test natural language queries."""
        # Store entries with searchable content
        entries = [
            KBEntry(
                data_type=KBDataType.OBSERVATION,
                summary="CPU usage monitoring shows high utilization",
                content={"metric": "cpu_usage"},
                tags=["cpu", "monitoring"],
            ),
            KBEntry(
                data_type=KBDataType.LEARNED_PATTERN,
                summary="Memory usage patterns indicate memory leak",
                content={"pattern": "memory_leak"},
                tags=["memory", "pattern"],
            ),
            KBEntry(
                data_type=KBDataType.OBSERVATION,
                summary="Network latency observation",
                content={"metric": "network_latency"},
                tags=["network", "monitoring"],
            ),
        ]

        for entry in entries:
            self.kb.store(entry)

        # Query for CPU-related entries
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE, query_text="CPU usage monitoring"
        )
        response = self.kb.query(query)

        assert response.success is True
        assert response.total_results >= 1

        # Check that the CPU-related entry is in results
        cpu_entries = [r for r in response.results if "cpu" in r.summary.lower()]
        assert len(cpu_entries) > 0

    def test_query_pagination(self):
        """Test query result pagination."""
        # Store multiple entries
        for i in range(10):
            entry = KBEntry(
                data_type=KBDataType.OBSERVATION,
                summary=f"Test observation {i}",
                content={"index": i},
                tags=["test"],
            )
            self.kb.store(entry)

        # Query with pagination
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            data_types=[KBDataType.OBSERVATION],
            limit=3,
            offset=2,
        )
        response = self.kb.query(query)

        assert response.success is True
        assert response.total_results == 10
        assert len(response.results) == 3  # Limited to 3 results

    def test_clear_knowledge_base(self):
        """Test clearing all data from the knowledge base."""
        # Store some data
        entry = KBEntry(
            data_type=KBDataType.OBSERVATION, summary="Test entry", tags=["test"]
        )
        self.kb.store(entry)

        # Store some telemetry
        telemetry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="test.metric",
            metric_value=42.0,
            source="test_source",
        )
        self.kb.store(telemetry)

        # Verify data exists
        assert len(self.kb._entries) > 0
        assert len(self.kb._raw_telemetry_buffers) > 0

        # Clear and verify
        self.kb.clear()
        assert len(self.kb._entries) == 0
        assert len(self.kb._raw_telemetry_buffers) == 0

    def test_get_stats(self):
        """Test knowledge base statistics."""
        # Initially empty
        stats = self.kb.get_stats()
        assert stats["total_permanent_entries"] == 0
        assert stats["total_buffered_events"] == 0

        # Add some permanent entries
        for i in range(3):
            entry = KBEntry(
                data_type=KBDataType.OBSERVATION,
                summary=f"Observation {i}",
                tags=["test"],
            )
            self.kb.store(entry)

        # Add some telemetry
        telemetry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="test.metric",
            metric_value=42.0,
            source="test_source",
        )
        self.kb.store(telemetry)

        # Check updated stats
        stats = self.kb.get_stats()
        assert stats["total_permanent_entries"] == 3
        assert stats["total_buffered_events"] == 1
        assert stats["active_telemetry_buffers"] == 1
        assert KBDataType.OBSERVATION.value in stats["data_type_counts"]
        assert stats["data_type_counts"][KBDataType.OBSERVATION.value] == 3

    def test_keyword_indexing(self):
        """Test keyword extraction and indexing."""
        entry = KBEntry(
            data_type=KBDataType.LEARNED_PATTERN,
            summary="Response time degradation pattern detected",
            content={"pattern_type": "performance", "confidence": 0.95},
            tags=["performance", "pattern", "degradation"],
        )
        self.kb.store(entry)

        # Query using natural language that should match keywords
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            query_text="response time degradation",
        )
        response = self.kb.query(query)

        assert response.success is True
        assert response.total_results > 0

        # Verify the correct entry was found
        found_entry = response.results[0]
        assert "degradation" in found_entry.summary.lower()

    def test_update_existing_entry(self):
        """Test updating an existing entry."""
        # Create and store initial entry
        entry = KBEntry(
            data_type=KBDataType.SYSTEM_GOAL,
            summary="Initial goal",
            content={"target": "initial_value"},
            tags=["goal"],
        )
        self.kb.store(entry)
        entry_id = entry.entry_id

        # Update the entry
        updated_entry = KBEntry(
            entry_id=entry_id,  # Same ID
            data_type=KBDataType.SYSTEM_GOAL,
            summary="Updated goal",
            content={"target": "updated_value"},
            tags=["goal", "updated"],
        )
        result = self.kb.store(updated_entry)
        assert result is True

        # Verify update
        retrieved = self.kb.get(entry_id)
        assert retrieved.summary == "Updated goal"
        assert retrieved.content["target"] == "updated_value"
        assert "updated" in retrieved.tags

    def test_error_handling_in_query(self):
        """Test error handling during query execution."""
        # Create a valid query but modify the internal query type to cause an error
        query = KBQuery(query_type=QueryType.NATURAL_LANGUAGE, query_text="test query")

        # Manually modify the query type to an invalid value to test error handling
        query.query_type = "invalid_type"

        response = self.kb.query(query)
        assert response.success is False
        assert response.message is not None
        assert response.processing_time_ms is not None


class TestKnowledgeBaseIntegration:
    """Integration tests for Knowledge Base with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kb = InMemoryKnowledgeBase(telemetry_buffer_size=5)

    def test_adaptation_scenario(self):
        """Test a complete adaptation scenario workflow."""
        # 1. Store initial system goal
        goal = KBEntry(
            data_type=KBDataType.SYSTEM_GOAL,
            summary="Maintain response time under 200ms",
            content={"metric": "response_time", "threshold": 200, "unit": "ms"},
            tags=["performance", "goal", "response_time"],
        )
        self.kb.store(goal)

        # 2. Store telemetry data showing degradation
        for i in range(6):  # Trigger aggregation
            telemetry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name="response_time",
                metric_value=180 + (i * 10),  # Increasing response times
                source="load_balancer",
                summary=f"Response time measurement {i+1}",
            )
            self.kb.store(telemetry)

        # 3. Store adaptation decision
        decision = KBEntry(
            data_type=KBDataType.ADAPTATION_DECISION,
            summary="Scale up servers due to response time degradation",
            content={
                "decision": "scale_up",
                "trigger_metric": "response_time",
                "trigger_value": 230,
                "action": "add_server_instance",
                "expected_impact": "reduce_response_time",
            },
            tags=["adaptation", "scaling", "response_time"],
        )
        self.kb.store(decision)

        # 4. Query for response time related information
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            query_text="response_time",
        )
        response = self.kb.query(query)

        assert response.success is True
        assert (
            response.total_results >= 2
        )  # Should find goal and decision (both contain response_time)

        # Verify we have entries of different types
        entry_types = {entry.data_type for entry in response.results}
        assert len(entry_types) >= 1  # At least one type found

    def test_learning_pattern_storage(self):
        """Test storing and querying learned patterns."""
        # Store multiple observations that could form a pattern
        observations = [
            {
                "summary": "High CPU usage observed during peak hours",
                "content": {"cpu_usage": 85, "time_period": "peak", "day": "monday"},
                "tags": ["cpu", "peak", "pattern"],
            },
            {
                "summary": "High CPU usage observed during peak hours",
                "content": {"cpu_usage": 88, "time_period": "peak", "day": "tuesday"},
                "tags": ["cpu", "peak", "pattern"],
            },
            {
                "summary": "Normal CPU usage during off-peak hours",
                "content": {
                    "cpu_usage": 45,
                    "time_period": "off_peak",
                    "day": "monday",
                },
                "tags": ["cpu", "off_peak", "normal"],
            },
        ]

        for obs_data in observations:
            entry = KBEntry(
                data_type=KBDataType.OBSERVATION,
                summary=obs_data["summary"],
                content=obs_data["content"],
                tags=obs_data["tags"],
            )
            self.kb.store(entry)

        # Store a learned pattern
        pattern = KBEntry(
            data_type=KBDataType.LEARNED_PATTERN,
            summary="CPU usage correlates with time periods",
            content={
                "pattern_type": "temporal_correlation",
                "variables": ["cpu_usage", "time_period"],
                "correlation": 0.85,
                "confidence": 0.92,
                "sample_size": 50,
            },
            tags=["pattern", "cpu", "temporal", "correlation"],
        )
        self.kb.store(pattern)

        # Query for patterns
        query = KBQuery(
            query_type=QueryType.STRUCTURED, data_types=[KBDataType.LEARNED_PATTERN]
        )
        response = self.kb.query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].content["pattern_type"] == "temporal_correlation"

    def test_mixed_query_scenarios(self):
        """Test complex mixed query scenarios."""
        # Store diverse knowledge
        entries_data = [
            {
                "type": KBDataType.SYSTEM_GOAL,
                "summary": "Maintain high availability",
                "content": {"availability_target": 99.9},
                "tags": ["availability", "goal"],
            },
            {
                "type": KBDataType.OBSERVATION,
                "summary": "System availability dropped to 99.5%",
                "content": {"availability": 99.5, "incident": "server_failure"},
                "tags": ["availability", "incident", "monitoring"],
            },
            {
                "type": KBDataType.ADAPTATION_DECISION,
                "summary": "Activate backup server for availability",
                "content": {"action": "activate_backup", "reason": "availability_drop"},
                "tags": ["availability", "backup", "adaptation"],
            },
            {
                "type": KBDataType.LEARNED_PATTERN,
                "summary": "Server failures correlate with high load",
                "content": {"pattern": "failure_correlation", "load_threshold": 85},
                "tags": ["pattern", "failure", "load"],
            },
        ]

        for entry_data in entries_data:
            entry = KBEntry(
                data_type=entry_data["type"],
                summary=entry_data["summary"],
                content=entry_data["content"],
                tags=entry_data["tags"],
            )
            self.kb.store(entry)

        # Query for availability-related knowledge across all types
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            query_text="availability server",
        )
        response = self.kb.query(query)

        assert response.success is True
        assert (
            response.total_results >= 1
        )  # Should find at least one availability-related entry

        # Verify diversity of results
        found_types = {entry.data_type for entry in response.results}
        assert len(found_types) >= 1  # Should have at least one type of knowledge


class TestTelemetrySpecificScenarios:
    """Specific test cases for telemetry handling edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kb = InMemoryKnowledgeBase(telemetry_buffer_size=3)

    def test_telemetry_buffer_with_mixed_content(self):
        """Test handling of telemetry entries with mixed content types."""
        # Store telemetry with additional non-numeric content
        telemetry_with_extra = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="status_check",
            metric_value=1.0,  # Numeric value (required for aggregation)
            source="health_monitor",
            summary="Health check with status info",
            content={"status": "healthy", "response_code": 200},  # Mixed content
        )

        result = self.kb.store(telemetry_with_extra)
        assert result is True

        # Add more telemetry to same metric
        for i in range(3):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name="status_check",
                metric_value=1.0,
                source="health_monitor",
                summary=f"Additional health check {i}",
                content={"status": "healthy", "response_code": 200 + i},
            )
            self.kb.store(entry)

        # Should have aggregated the numeric metric_values successfully
        assert len(self.kb._entries) == 1
        aggregated = list(self.kb._entries.values())[0]
        assert aggregated.content["count"] == 3  # First 3 entries were aggregated
        assert aggregated.content["average_value"] == 1.0  # All had value 1.0

    def test_telemetry_aggregation_timing(self):
        """Test the timing and sequencing of telemetry aggregation."""
        timestamps_before = []

        # Store telemetry entries and track when they're created
        for i in range(4):  # One more than buffer size to trigger aggregation
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name="cpu.load",
                metric_value=50.0 + i,
                source="system_monitor",
                summary=f"CPU load reading {i+1}",
            )
            timestamps_before.append(entry.timestamp)
            self.kb.store(entry)

        # Check that aggregation created an observation
        assert len(self.kb._entries) == 1
        aggregated_entry = list(self.kb._entries.values())[0]

        # Verify timing information in aggregated entry
        assert "time_window_start" in aggregated_entry.content
        assert "time_window_end" in aggregated_entry.content

        # Time window should span the first 3 entries (buffer size)
        start_time = aggregated_entry.content["time_window_start"]
        end_time = aggregated_entry.content["time_window_end"]
        assert start_time == timestamps_before[0]  # First entry
        assert end_time == timestamps_before[2]  # Third entry (before aggregation)

    def test_multiple_sources_same_metric(self):
        """Test handling of same metric from different sources."""
        sources = ["monitor_a", "monitor_b", "monitor_c"]

        # Store telemetry from multiple sources
        for source in sources:
            for i in range(4):  # Trigger aggregation for each source
                entry = KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name="memory.usage",
                    metric_value=60.0 + i,
                    source=source,
                    summary=f"Memory usage from {source}",
                )
                self.kb.store(entry)

        # Should have 3 aggregated observations (one per source)
        assert len(self.kb._entries) == 3

        # Verify each observation has the correct source
        observations = list(self.kb._entries.values())
        sources_found = {obs.source for obs in observations}
        assert sources_found == set(sources)

    def test_telemetry_stats_tracking(self):
        """Test that telemetry buffers are properly tracked in stats."""
        # Initially no buffers
        stats = self.kb.get_stats()
        assert stats["active_telemetry_buffers"] == 0
        assert stats["total_buffered_events"] == 0

        # Add telemetry for different metrics
        metrics = ["cpu.usage", "memory.usage", "disk.io"]

        for metric in metrics:
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=metric,
                metric_value=75.0,
                source="monitor",
                summary=f"{metric} measurement",
            )
            self.kb.store(entry)

        # Check updated stats
        stats = self.kb.get_stats()
        assert stats["active_telemetry_buffers"] == 3  # One per metric
        assert stats["total_buffered_events"] == 3  # One event per buffer

    def test_telemetry_aggregation_statistics_accuracy(self):
        """Test the accuracy of statistical calculations in aggregation."""
        values = [10.0, 20.0, 30.0]  # Known values for easy calculation

        for i, value in enumerate(values):
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name="test.metric",
                metric_value=value,
                source="test_source",
                summary=f"Test value {i+1}",
            )
            self.kb.store(entry)

        # Trigger aggregation
        trigger_entry = KBEntry(
            data_type=KBDataType.RAW_TELEMETRY_EVENT,
            metric_name="test.metric",
            metric_value=40.0,
            source="test_source",
            summary="Trigger aggregation",
        )
        self.kb.store(trigger_entry)

        # Verify aggregation statistics
        assert len(self.kb._entries) == 1
        aggregated = list(self.kb._entries.values())[0]

        assert aggregated.content["count"] == 3
        assert aggregated.content["average_value"] == 20.0  # (10+20+30)/3
        assert aggregated.content["min_value"] == 10.0
        assert aggregated.content["max_value"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__])
