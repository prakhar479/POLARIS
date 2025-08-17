"""
Integration tests for Knowledge Base System.

Tests the integration between query models, storage, and end-to-end workflows
in the POLARIS Knowledge Base system.
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

from polaris.common.query_models import QueryType, KBQuery, KBEntry, KBResponse
from polaris.common.kb_storage import InMemoryKBStorage


class MockKnowledgeBaseService:
    """Mock Knowledge Base service for integration testing."""

    def __init__(self, storage=None, logger=None):
        self.storage = storage or InMemoryKBStorage(logger)
        self.logger = logger or logging.getLogger(__name__)
        self._query_count = 0

    def process_query(self, query: KBQuery) -> KBResponse:
        """Process a knowledge base query and return response."""
        start_time = time.time()
        self._query_count += 1

        try:
            if query.query_type == QueryType.STRUCTURED:
                results = self._process_structured_query(query)
            elif query.query_type == QueryType.NATURAL_LANGUAGE:
                results = self._process_natural_language_query(query)
            elif query.query_type == QueryType.SEMANTIC:
                results = self._process_semantic_query(query)
            else:
                return KBResponse(
                    query_id=query.query_id,
                    success=False,
                    message=f"Unsupported query type: {query.query_type}",
                )

            processing_time = (time.time() - start_time) * 1000

            return KBResponse(
                query_id=query.query_id,
                success=True,
                results=results,
                total_results=len(results),
                processing_time_ms=processing_time,
                metadata={
                    "query_type": query.query_type,
                    "filters_applied": query.filters is not None,
                    "query_number": self._query_count,
                },
            )

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            processing_time = (time.time() - start_time) * 1000

            return KBResponse(
                query_id=query.query_id,
                success=False,
                message=f"Query processing failed: {str(e)}",
                processing_time_ms=processing_time,
            )

    def _process_structured_query(self, query: KBQuery):
        """Process structured query using filters."""
        if query.filters:
            return self.storage.search_by_filters(query.filters)
        else:
            return self.storage.search_by_content(query.content)

    def _process_natural_language_query(self, query: KBQuery):
        """Process natural language query with simple text search."""
        # Simple implementation - in real system would use NLP
        return self.storage.search_by_content(query.content)

    def _process_semantic_query(self, query: KBQuery):
        """Process semantic query (simplified for testing)."""
        # Simple implementation - in real system would use embeddings
        results = self.storage.search_by_content(query.content)

        # Apply context filters if available
        if query.context and "category" in query.context:
            category = query.context["category"]
            results = [r for r in results if r.content.get("category") == category]

        return results

    def add_entry(self, entry: KBEntry) -> bool:
        """Add entry to knowledge base."""
        return self.storage.store(entry)

    def update_entry(self, entry_id: str, updates: dict) -> bool:
        """Update existing entry."""
        return self.storage.update(entry_id, updates)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete entry from knowledge base."""
        return self.storage.delete(entry_id)

    def get_statistics(self):
        """Get system statistics."""
        storage_stats = self.storage.get_stats()
        return {**storage_stats, "total_queries_processed": self._query_count}


class TestKnowledgeBaseIntegration:
    """Test complete knowledge base system integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.kb_service = MockKnowledgeBaseService(logger=self.logger)

        # Add sample data
        self._setup_sample_data()

    def _setup_sample_data(self):
        """Set up sample knowledge base entries."""
        sample_entries = [
            KBEntry(
                entry_id="sys-error-001",
                content={
                    "title": "Critical System Error",
                    "description": "Database connection pool exhausted",
                    "severity": "critical",
                    "category": "database",
                    "solution": "Restart database service and increase pool size",
                },
                tags=["error", "database", "critical", "connection"],
                metadata={"reported_by": "monitor", "incident_id": "INC-001"},
            ),
            KBEntry(
                entry_id="perf-warn-001",
                content={
                    "title": "High CPU Usage",
                    "description": "CPU usage exceeded 85% for 5 minutes",
                    "severity": "warning",
                    "category": "performance",
                    "solution": "Check for resource-intensive processes",
                },
                tags=["warning", "performance", "cpu", "monitoring"],
                metadata={"reported_by": "monitor", "threshold": "85%"},
            ),
            KBEntry(
                entry_id="net-info-001",
                content={
                    "title": "Network Configuration Updated",
                    "description": "Updated network routing configuration",
                    "severity": "info",
                    "category": "network",
                    "solution": "Monitor for connectivity issues",
                },
                tags=["info", "network", "configuration", "routing"],
                metadata={"reported_by": "admin", "change_id": "CHG-001"},
            ),
        ]

        for entry in sample_entries:
            self.kb_service.add_entry(entry)

    def test_structured_query_integration(self):
        """Test structured query processing end-to-end."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="find critical errors",
            filters={"severity": "critical", "category": "database"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.query_id == query.query_id
        assert response.total_results == 1
        assert len(response.results) == 1
        assert response.results[0].entry_id == "sys-error-001"
        assert response.processing_time_ms > 0
        assert response.metadata["query_type"] == QueryType.STRUCTURED
        assert response.metadata["filters_applied"] is True

    def test_natural_language_query_integration(self):
        """Test natural language query processing."""
        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            content="CPU usage",  # Changed to match actual content
            context={"user": "operator"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].entry_id == "perf-warn-001"
        assert "CPU" in response.results[0].content["title"]

    def test_semantic_query_integration(self):
        """Test semantic query processing with context."""
        query = KBQuery(
            query_type=QueryType.SEMANTIC,
            content="configuration",  # Changed to match actual content
            context={"category": "network"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].entry_id == "net-info-001"
        assert response.results[0].content["category"] == "network"

    def test_query_with_no_results(self):
        """Test query that returns no results."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="find missing data",
            filters={"severity": "nonexistent"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 0
        assert len(response.results) == 0
        assert response.processing_time_ms > 0

    def test_invalid_query_type(self):
        """Test handling of invalid query type."""

        # Create a mock query object that bypasses validation
        class MockQuery:
            def __init__(self):
                self.query_id = str(uuid.uuid4())
                self.query_type = "invalid_type"
                self.content = "test query"
                self.filters = None
                self.context = None
                self.timestamp = datetime.now(timezone.utc).isoformat()

        mock_query = MockQuery()
        response = self.kb_service.process_query(mock_query)

        assert response.success is False
        assert "Unsupported query type" in response.message

    def test_multiple_queries_performance(self):
        """Test processing multiple queries and performance tracking."""
        queries = [
            KBQuery(
                query_type=QueryType.STRUCTURED,
                content="errors",
                filters={"severity": "critical"},
            ),
            KBQuery(
                query_type=QueryType.NATURAL_LANGUAGE, content="performance warnings"
            ),
            KBQuery(query_type=QueryType.SEMANTIC, content="network updates"),
        ]

        responses = []
        for query in queries:
            response = self.kb_service.process_query(query)
            responses.append(response)

        # Verify all queries succeeded
        assert all(r.success for r in responses)

        # Verify query numbering
        assert responses[0].metadata["query_number"] == 1
        assert responses[1].metadata["query_number"] == 2
        assert responses[2].metadata["query_number"] == 3

        # Verify statistics
        stats = self.kb_service.get_statistics()
        assert stats["total_queries_processed"] == 3
        assert stats["total_entries"] == 3

    def test_filter_only_query_integration(self):
        """Test filter-only queries work end-to-end."""
        # Create a filter-only query (empty content)
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="",  # Empty content
            filters={"category": "database"},  # Changed from "system" to "database"
            context={"user": "admin"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].entry_id == "sys-error-001"
        assert response.results[0].content["category"] == "database"

    def test_whitespace_only_query_with_filters(self):
        """Test queries with only whitespace content but valid filters."""
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="   \t\n   ",  # Only whitespace
            filters={"severity": "critical"},  # Changed from "high" to "critical"
            context={"user": "operator"},
        )

        response = self.kb_service.process_query(query)

        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].content["severity"] == "critical"


class TestKnowledgeBaseCRUDIntegration:
    """Test CRUD operations integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kb_service = MockKnowledgeBaseService()

    def test_complete_entry_lifecycle_integration(self):
        """Test complete entry lifecycle through service."""
        # Create new entry
        entry = KBEntry(
            entry_id="lifecycle-test",
            content={
                "title": "Test Integration Entry",
                "description": "Testing complete lifecycle",
                "category": "test",
            },
            tags=["test", "integration"],
        )

        # Add entry
        assert self.kb_service.add_entry(entry) is True

        # Query for the entry
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="test",
            filters={"category": "test"},
        )

        response = self.kb_service.process_query(query)
        assert response.success is True
        assert response.total_results == 1
        assert response.results[0].entry_id == "lifecycle-test"

        # Update entry
        updates = {
            "content": {"status": "updated"},
            "tags": ["test", "integration", "updated"],
        }
        assert self.kb_service.update_entry("lifecycle-test", updates) is True

        # Query updated entry
        response = self.kb_service.process_query(query)
        updated_entry = response.results[0]
        assert updated_entry.content["status"] == "updated"
        assert "updated" in updated_entry.tags

        # Delete entry
        assert self.kb_service.delete_entry("lifecycle-test") is True

        # Verify deletion
        response = self.kb_service.process_query(query)
        assert response.total_results == 0

    def test_batch_operations_integration(self):
        """Test batch operations on multiple entries."""
        # Create multiple entries
        entries = []
        for i in range(5):
            entry = KBEntry(
                entry_id=f"batch-{i}",
                content={
                    "title": f"Batch Entry {i}",
                    "index": i,
                    "category": "batch_test",
                },
                tags=["batch", f"item{i}"],
            )
            entries.append(entry)
            assert self.kb_service.add_entry(entry) is True

        # Query all batch entries
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="batch entries",
            filters={"category": "batch_test"},
        )

        response = self.kb_service.process_query(query)
        assert response.success is True
        assert response.total_results == 5

        # Update all entries
        for i in range(5):
            updates = {"content": {"updated": True}}
            assert self.kb_service.update_entry(f"batch-{i}", updates) is True

        # Verify updates
        response = self.kb_service.process_query(query)
        for result in response.results:
            assert result.content.get("updated") is True

        # Delete half the entries
        for i in range(3):
            assert self.kb_service.delete_entry(f"batch-{i}") is True

        # Verify partial deletion
        response = self.kb_service.process_query(query)
        assert response.total_results == 2


class TestKnowledgeBaseErrorHandling:
    """Test error handling in integrated system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)

        # Create storage that might fail
        self.failing_storage = Mock(spec=InMemoryKBStorage)
        self.kb_service = MockKnowledgeBaseService(
            storage=self.failing_storage, logger=self.logger
        )

    def test_storage_failure_handling(self):
        """Test handling of storage failures."""
        # Mock storage to raise exception
        self.failing_storage.search_by_filters.side_effect = Exception("Storage failed")

        query = KBQuery(
            query_type=QueryType.STRUCTURED, content="test", filters={"test": "value"}
        )

        response = self.kb_service.process_query(query)

        assert response.success is False
        assert "Query processing failed" in response.message
        assert "Storage failed" in response.message
        assert response.processing_time_ms > 0

        # Verify error was logged
        self.logger.error.assert_called_once()

    def test_query_processing_resilience(self):
        """Test system resilience during query processing."""

        # Set up partial failure scenario
        def side_effect(*args, **kwargs):
            # Succeed on first call, fail on second
            if not hasattr(side_effect, "call_count"):
                side_effect.call_count = 0
            side_effect.call_count += 1

            if side_effect.call_count == 1:
                return []  # Success with no results
            else:
                raise Exception("Intermittent failure")

        self.failing_storage.search_by_filters.side_effect = side_effect

        # First query succeeds
        query1 = KBQuery(
            query_type=QueryType.STRUCTURED, content="test1", filters={"test": "value1"}
        )

        response1 = self.kb_service.process_query(query1)
        assert response1.success is True
        assert response1.total_results == 0

        # Second query fails
        query2 = KBQuery(
            query_type=QueryType.STRUCTURED, content="test2", filters={"test": "value2"}
        )

        response2 = self.kb_service.process_query(query2)
        assert response2.success is False
        assert "Intermittent failure" in response2.message


class TestKnowledgeBasePerformance:
    """Test performance characteristics of integrated system."""

    def setup_method(self):
        """Set up test fixtures with larger dataset."""
        self.kb_service = MockKnowledgeBaseService()

        # Create larger dataset
        for i in range(100):
            entry = KBEntry(
                entry_id=f"perf-entry-{i:03d}",
                content={
                    "title": f"Performance Entry {i}",
                    "index": i,
                    "category": f"cat{i % 10}",
                    "priority": "high" if i % 10 == 0 else "normal",
                    "data": f"Large data content for entry {i}" * 10,
                },
                tags=[f"tag{i % 5}", f"group{i % 3}", "performance"],
            )
            self.kb_service.add_entry(entry)

    def test_query_performance_characteristics(self):
        """Test query performance with larger dataset."""
        # Test different query types
        queries = [
            KBQuery(
                query_type=QueryType.STRUCTURED,
                content="structured query",
                filters={"priority": "high"},
            ),
            KBQuery(query_type=QueryType.NATURAL_LANGUAGE, content="performance data"),
            KBQuery(
                query_type=QueryType.SEMANTIC,
                content="category search",
                context={"category": "cat1"},
            ),
        ]

        processing_times = []
        for query in queries:
            response = self.kb_service.process_query(query)
            assert response.success is True
            processing_times.append(response.processing_time_ms)

        # Verify reasonable processing times (under 100ms for this dataset)
        for time_ms in processing_times:
            assert time_ms < 100, f"Query took too long: {time_ms}ms"

        # Verify statistics
        stats = self.kb_service.get_statistics()
        assert stats["total_entries"] == 100
        assert stats["total_queries_processed"] == len(queries)

    def test_concurrent_query_simulation(self):
        """Test simulated concurrent query processing."""
        # Simulate multiple concurrent queries
        queries = []
        for i in range(20):
            query = KBQuery(
                query_type=QueryType.STRUCTURED,
                content=f"concurrent query {i}",
                filters={"category": f"cat{i % 10}"},
            )
            queries.append(query)

        # Process all queries
        responses = []
        start_time = time.time()

        for query in queries:
            response = self.kb_service.process_query(query)
            responses.append(response)

        total_time = time.time() - start_time

        # Verify all succeeded
        assert all(r.success for r in responses)

        # Verify reasonable total processing time
        assert total_time < 1.0, f"Total processing took too long: {total_time}s"

        # Verify query numbering is correct
        for i, response in enumerate(responses):
            assert response.metadata["query_number"] == i + 1


if __name__ == "__main__":
    pytest.main([__file__])
