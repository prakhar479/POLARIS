"""
Unit tests for Knowledge Base Query Models.

Tests the query models, responses, and data structures used
in the POLARIS Knowledge Base system.
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

from polaris.common.query_models import QueryType, KBQuery, KBEntry, KBResponse


class TestQueryType:
    """Test cases for QueryType enum."""

    def test_query_type_values(self):
        """Test QueryType enum values."""
        assert QueryType.STRUCTURED == "structured"
        assert QueryType.NATURAL_LANGUAGE == "natural_language"
        assert QueryType.SEMANTIC == "semantic"

    def test_query_type_membership(self):
        """Test QueryType membership checks."""
        assert QueryType.STRUCTURED in QueryType
        assert QueryType.NATURAL_LANGUAGE in QueryType
        assert QueryType.SEMANTIC in QueryType
        # Test values
        assert QueryType.STRUCTURED.value == "structured"
        assert QueryType.NATURAL_LANGUAGE.value == "natural_language"
        assert QueryType.SEMANTIC.value == "semantic"


class TestKBQuery:
    """Test cases for KBQuery model."""

    def test_kb_query_creation_minimal(self):
        """Test creating KBQuery with minimal required fields."""
        query = KBQuery(query_type=QueryType.STRUCTURED, content="test query")

        assert query.query_type == QueryType.STRUCTURED
        assert query.content == "test query"
        assert query.query_id is not None
        assert query.timestamp is not None
        assert query.filters is None
        assert query.context is None

    def test_kb_query_creation_full(self):
        """Test creating KBQuery with all fields."""
        filters = {"category": "system", "priority": "high"}
        context = {"user": "admin", "session": "123"}

        query = KBQuery(
            query_type=QueryType.NATURAL_LANGUAGE,
            content="Find all high priority system issues",
            filters=filters,
            context=context,
        )

        assert query.query_type == QueryType.NATURAL_LANGUAGE
        assert query.content == "Find all high priority system issues"
        assert query.filters == filters
        assert query.context == context
        assert query.query_id is not None
        assert query.timestamp is not None

    def test_kb_query_id_generation(self):
        """Test that query IDs are unique and valid UUIDs."""
        query1 = KBQuery(query_type=QueryType.STRUCTURED, content="test1")
        query2 = KBQuery(query_type=QueryType.STRUCTURED, content="test2")

        # Should be different
        assert query1.query_id != query2.query_id

        # Should be valid UUID strings
        uuid.UUID(query1.query_id)  # Will raise if invalid
        uuid.UUID(query2.query_id)  # Will raise if invalid

    def test_kb_query_timestamp_format(self):
        """Test that timestamps are in ISO format."""
        query = KBQuery(query_type=QueryType.STRUCTURED, content="test")

        # Should be parseable as ISO format
        parsed_time = datetime.fromisoformat(query.timestamp.replace("Z", "+00:00"))
        assert parsed_time.tzinfo is not None

    def test_kb_query_serialization(self):
        """Test KBQuery serialization to dict."""
        filters = {"type": "error"}
        context = {"source": "monitor"}

        query = KBQuery(
            query_type=QueryType.SEMANTIC,
            content="semantic search test",
            filters=filters,
            context=context,
        )

        query_dict = query.model_dump()

        assert query_dict["query_type"] == "semantic"
        assert query_dict["content"] == "semantic search test"
        assert query_dict["filters"] == filters
        assert query_dict["context"] == context
        assert "query_id" in query_dict
        assert "timestamp" in query_dict

    def test_kb_query_validation(self):
        """Test KBQuery field validation."""
        # Test invalid query type
        with pytest.raises(ValueError):
            KBQuery(query_type="invalid_type", content="test")

        # Test empty content
        with pytest.raises(ValueError):
            KBQuery(query_type=QueryType.STRUCTURED, content="")


class TestKBEntry:
    """Test cases for KBEntry model."""

    def test_kb_entry_creation_minimal(self):
        """Test creating KBEntry with minimal required fields."""
        content = {"title": "Test Entry", "body": "Test content"}
        entry = KBEntry(entry_id="entry-123", content=content)

        assert entry.entry_id == "entry-123"
        assert entry.content == content
        assert entry.tags is None
        assert entry.metadata is None
        assert entry.created_at is not None
        assert entry.updated_at is not None

    def test_kb_entry_creation_full(self):
        """Test creating KBEntry with all fields."""
        content = {
            "title": "System Error Analysis",
            "description": "Analysis of system errors",
            "severity": "high",
        }
        tags = ["error", "system", "analysis"]
        metadata = {"source": "monitor", "category": "diagnostics", "version": "1.0"}

        entry = KBEntry(
            entry_id="entry-456", content=content, tags=tags, metadata=metadata
        )

        assert entry.entry_id == "entry-456"
        assert entry.content == content
        assert entry.tags == tags
        assert entry.metadata == metadata
        assert entry.created_at is not None
        assert entry.updated_at is not None

    def test_kb_entry_timestamp_format(self):
        """Test that entry timestamps are in ISO format."""
        entry = KBEntry(entry_id="test", content={"test": "data"})

        # Should be parseable as ISO format
        created_time = datetime.fromisoformat(entry.created_at.replace("Z", "+00:00"))
        updated_time = datetime.fromisoformat(entry.updated_at.replace("Z", "+00:00"))

        assert created_time.tzinfo is not None
        assert updated_time.tzinfo is not None

    def test_kb_entry_serialization(self):
        """Test KBEntry serialization to dict."""
        content = {"key": "value", "nested": {"data": 123}}
        tags = ["tag1", "tag2"]
        metadata = {"meta": "info"}

        entry = KBEntry(
            entry_id="serialize-test", content=content, tags=tags, metadata=metadata
        )

        entry_dict = entry.model_dump()

        assert entry_dict["entry_id"] == "serialize-test"
        assert entry_dict["content"] == content
        assert entry_dict["tags"] == tags
        assert entry_dict["metadata"] == metadata
        assert "created_at" in entry_dict
        assert "updated_at" in entry_dict

    def test_kb_entry_json_serialization(self):
        """Test KBEntry JSON serialization."""
        content = {"data": [1, 2, 3], "info": "test"}
        entry = KBEntry(entry_id="json-test", content=content)

        # Should be JSON serializable
        json_str = entry.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["entry_id"] == "json-test"
        assert parsed["content"] == content


class TestKBResponse:
    """Test cases for KBResponse model."""

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

    def test_kb_response_creation_with_results(self):
        """Test creating KBResponse with results."""
        # Create some test entries
        entries = [
            KBEntry(entry_id="result-1", content={"title": "Result 1"}, tags=["test"]),
            KBEntry(entry_id="result-2", content={"title": "Result 2"}, tags=["test"]),
        ]

        response = KBResponse(
            query_id="query-456",
            success=True,
            results=entries,
            total_results=2,
            message="Found 2 matching entries",
            processing_time_ms=45.2,
        )

        assert response.query_id == "query-456"
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.message == "Found 2 matching entries"
        assert response.processing_time_ms == 45.2

    def test_kb_response_error_case(self):
        """Test creating KBResponse for error cases."""
        response = KBResponse(
            query_id="query-error",
            success=False,
            message="Query parsing failed",
            metadata={"error_code": "PARSE_ERROR"},
        )

        assert response.query_id == "query-error"
        assert response.success is False
        assert response.results == []
        assert response.total_results == 0
        assert response.message == "Query parsing failed"
        assert response.metadata["error_code"] == "PARSE_ERROR"

    def test_kb_response_serialization(self):
        """Test KBResponse serialization."""
        # Create test entry
        entry = KBEntry(entry_id="test-entry", content={"data": "test"})

        metadata = {"query_type": "structured", "filters_applied": True}

        response = KBResponse(
            query_id="serialize-test",
            success=True,
            results=[entry],
            total_results=1,
            message="Success",
            metadata=metadata,
            processing_time_ms=12.5,
        )

        response_dict = response.model_dump()

        assert response_dict["query_id"] == "serialize-test"
        assert response_dict["success"] is True
        assert len(response_dict["results"]) == 1
        assert response_dict["total_results"] == 1
        assert response_dict["message"] == "Success"
        assert response_dict["metadata"] == metadata
        assert response_dict["processing_time_ms"] == 12.5

    def test_kb_response_with_empty_results(self):
        """Test KBResponse with no results."""
        response = KBResponse(
            query_id="empty-results",
            success=True,
            message="No matching entries found",
            processing_time_ms=5.0,
        )

        assert response.success is True
        assert response.results == []
        assert response.total_results == 0
        assert response.message == "No matching entries found"


class TestModelIntegration:
    """Test integration between different models."""

    def test_query_response_integration(self):
        """Test complete query-response flow."""
        # Create query
        query = KBQuery(
            query_type=QueryType.STRUCTURED,
            content="find errors",
            filters={"severity": "high"},
            context={"user": "admin"},
        )

        # Create matching entries
        entries = [
            KBEntry(
                entry_id="error-1",
                content={
                    "title": "Critical System Error",
                    "severity": "high",
                    "description": "Database connection failed",
                },
                tags=["error", "database", "critical"],
            )
        ]

        # Create response
        response = KBResponse(
            query_id=query.query_id,
            success=True,
            results=entries,
            total_results=1,
            processing_time_ms=25.7,
        )

        assert response.query_id == query.query_id
        assert len(response.results) == 1
        assert response.results[0].content["severity"] == "high"

    def test_model_field_validation(self):
        """Test model field validation across all models."""
        # Test required fields
        with pytest.raises(ValueError):
            KBQuery()  # Missing required fields

        with pytest.raises(ValueError):
            KBEntry()  # Missing required fields

        with pytest.raises(ValueError):
            KBResponse()  # Missing required fields

    def test_model_defaults(self):
        """Test model default values."""
        # KBQuery defaults
        query = KBQuery(query_type=QueryType.STRUCTURED, content="test")
        assert query.filters is None
        assert query.context is None
        assert query.query_id is not None
        assert query.timestamp is not None

        # KBEntry defaults
        entry = KBEntry(entry_id="test", content={"test": "data"})
        assert entry.tags is None
        assert entry.metadata is None
        assert entry.created_at is not None
        assert entry.updated_at is not None

        # KBResponse defaults
        response = KBResponse(query_id="test", success=True)
        assert response.results == []
        assert response.total_results == 0
        assert response.message is None
        assert response.metadata is None
        assert response.processing_time_ms is None


if __name__ == "__main__":
    pytest.main([__file__])
