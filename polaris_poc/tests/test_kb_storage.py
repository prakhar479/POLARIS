"""
Unit tests for Knowledge Base Storage Implementation.

Tests the in-memory storage backend for the POLARIS Knowledge Base,
including CRUD operations, search functionality, and error handling.
"""

import pytest
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.kb_storage import InMemoryKBStorage
from polaris.common.query_models import KBEntry


class TestInMemoryKBStorageBasics:
    """Test basic functionality of InMemoryKBStorage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.storage = InMemoryKBStorage(logger=self.logger)

    def test_storage_initialization(self):
        """Test storage initializes correctly."""
        assert self.storage.logger is not None
        assert self.storage._entries == {}
        assert self.storage._tag_index == {}

    def test_storage_initialization_without_logger(self):
        """Test storage initializes with default logger."""
        storage = InMemoryKBStorage()
        assert storage.logger is not None
        assert hasattr(storage.logger, "debug")
        assert hasattr(storage.logger, "error")

    def test_empty_storage_stats(self):
        """Test statistics for empty storage."""
        stats = self.storage.get_stats()

        assert stats["total_entries"] == 0
        assert stats["total_tags"] == 0
        assert "memory_usage_estimate_kb" in stats
        assert stats["memory_usage_estimate_kb"] >= 0


class TestKBStorageCRUDOperations:
    """Test CRUD operations for KB storage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()
        self.test_entry = KBEntry(
            entry_id="test-entry-1",
            content={
                "title": "Test Entry",
                "description": "A test knowledge base entry",
                "category": "testing",
            },
            tags=["test", "example"],
            metadata={"source": "unit_test", "priority": "low"},
        )

    def test_store_entry_success(self):
        """Test successfully storing an entry."""
        result = self.storage.store(self.test_entry)

        assert result is True
        assert "test-entry-1" in self.storage._entries
        assert self.storage._entries["test-entry-1"] == self.test_entry

        # Check tag index
        assert "test" in self.storage._tag_index
        assert "example" in self.storage._tag_index
        assert "test-entry-1" in self.storage._tag_index["test"]
        assert "test-entry-1" in self.storage._tag_index["example"]

    def test_store_entry_updates_timestamp(self):
        """Test that storing updates the updated_at timestamp."""
        original_updated_at = self.test_entry.updated_at

        # Store entry
        self.storage.store(self.test_entry)

        # Timestamp should be updated
        stored_entry = self.storage._entries["test-entry-1"]
        assert stored_entry.updated_at != original_updated_at

    def test_store_entry_without_tags(self):
        """Test storing entry without tags."""
        entry = KBEntry(entry_id="no-tags", content={"data": "test"})

        result = self.storage.store(entry)

        assert result is True
        assert "no-tags" in self.storage._entries
        # No tags should be added to index
        assert len(self.storage._tag_index) == 0

    def test_get_existing_entry(self):
        """Test retrieving an existing entry."""
        self.storage.store(self.test_entry)

        retrieved = self.storage.get("test-entry-1")

        assert retrieved is not None
        assert retrieved.entry_id == "test-entry-1"
        assert retrieved.content == self.test_entry.content
        assert retrieved.tags == self.test_entry.tags

    def test_get_nonexistent_entry(self):
        """Test retrieving a nonexistent entry."""
        retrieved = self.storage.get("nonexistent")
        assert retrieved is None

    def test_update_entry_content(self):
        """Test updating entry content."""
        self.storage.store(self.test_entry)

        updates = {"content": {"new_field": "new_value"}}

        result = self.storage.update("test-entry-1", updates)

        assert result is True
        updated_entry = self.storage.get("test-entry-1")
        assert "new_field" in updated_entry.content
        assert updated_entry.content["new_field"] == "new_value"
        # Original content should be preserved and merged
        assert "title" in updated_entry.content

    def test_update_entry_tags(self):
        """Test updating entry tags."""
        self.storage.store(self.test_entry)

        new_tags = ["updated", "modified"]
        updates = {"tags": new_tags}

        result = self.storage.update("test-entry-1", updates)

        assert result is True
        updated_entry = self.storage.get("test-entry-1")
        assert updated_entry.tags == new_tags

        # Check tag index updates
        assert "updated" in self.storage._tag_index
        assert "modified" in self.storage._tag_index
        assert "test" not in self.storage._tag_index  # Old tags removed
        assert "example" not in self.storage._tag_index

    def test_update_entry_metadata(self):
        """Test updating entry metadata."""
        self.storage.store(self.test_entry)

        updates = {"metadata": {"new_meta": "value", "priority": "high"}}

        result = self.storage.update("test-entry-1", updates)

        assert result is True
        updated_entry = self.storage.get("test-entry-1")
        assert updated_entry.metadata["new_meta"] == "value"
        assert updated_entry.metadata["priority"] == "high"
        assert updated_entry.metadata["source"] == "unit_test"  # Preserved

    def test_update_nonexistent_entry(self):
        """Test updating a nonexistent entry."""
        result = self.storage.update("nonexistent", {"content": {"test": "data"}})
        assert result is False

    def test_delete_existing_entry(self):
        """Test deleting an existing entry."""
        self.storage.store(self.test_entry)

        result = self.storage.delete("test-entry-1")

        assert result is True
        assert "test-entry-1" not in self.storage._entries

        # Check tag index cleanup
        assert "test" not in self.storage._tag_index
        assert "example" not in self.storage._tag_index

    def test_delete_nonexistent_entry(self):
        """Test deleting a nonexistent entry."""
        result = self.storage.delete("nonexistent")
        assert result is False

    def test_clear_storage(self):
        """Test clearing all storage."""
        self.storage.store(self.test_entry)

        # Verify entry exists
        assert len(self.storage._entries) == 1
        assert len(self.storage._tag_index) == 2

        self.storage.clear()

        # Verify storage is empty
        assert len(self.storage._entries) == 0
        assert len(self.storage._tag_index) == 0


class TestKBStorageSearchOperations:
    """Test search functionality of KB storage."""

    def setup_method(self):
        """Set up test fixtures with multiple entries."""
        self.storage = InMemoryKBStorage()

        # Create test entries
        self.entries = [
            KBEntry(
                entry_id="error-1",
                content={
                    "title": "Database Connection Error",
                    "severity": "high",
                    "category": "database",
                    "description": "Unable to connect to primary database",
                },
                tags=["error", "database", "critical"],
            ),
            KBEntry(
                entry_id="warning-1",
                content={
                    "title": "High CPU Usage Warning",
                    "severity": "medium",
                    "category": "performance",
                    "description": "CPU usage exceeded 80% threshold",
                },
                tags=["warning", "performance", "cpu"],
            ),
            KBEntry(
                entry_id="info-1",
                content={
                    "title": "System Startup Complete",
                    "severity": "low",
                    "category": "system",
                    "description": "All services started successfully",
                },
                tags=["info", "system", "startup"],
            ),
        ]

        # Store all entries
        for entry in self.entries:
            self.storage.store(entry)

    def test_search_by_single_filter(self):
        """Test searching with a single filter."""
        results = self.storage.search_by_filters({"severity": "high"})

        assert len(results) == 1
        assert results[0].entry_id == "error-1"
        assert results[0].content["severity"] == "high"

    def test_search_by_multiple_filters(self):
        """Test searching with multiple filters."""
        results = self.storage.search_by_filters(
            {"category": "performance", "severity": "medium"}
        )

        assert len(results) == 1
        assert results[0].entry_id == "warning-1"

    def test_search_by_tags_filter(self):
        """Test searching by tags filter."""
        results = self.storage.search_by_filters({"tags": ["error"]})

        assert len(results) == 1
        assert results[0].entry_id == "error-1"

        # Test multiple tags
        results = self.storage.search_by_filters({"tags": ["performance", "system"]})

        assert len(results) == 2
        result_ids = {r.entry_id for r in results}
        assert "warning-1" in result_ids
        assert "info-1" in result_ids

    def test_search_by_metadata_filter(self):
        """Test searching by metadata filters."""
        # Add metadata to one entry
        self.storage.update(
            "error-1", {"metadata": {"source": "monitor", "alert_id": "ALT-001"}}
        )

        # Search by metadata field that should only match one entry
        results = self.storage.search_by_filters({"alert_id": "ALT-001"})

        assert len(results) == 1
        assert results[0].entry_id == "error-1"

    def test_search_no_matches(self):
        """Test searching with no matching results."""
        results = self.storage.search_by_filters({"severity": "nonexistent"})
        assert len(results) == 0

    def test_search_by_content_text(self):
        """Test text search in content."""
        results = self.storage.search_by_content("database")

        assert len(results) == 1
        assert results[0].entry_id == "error-1"

    def test_search_by_content_case_insensitive(self):
        """Test case-insensitive content search."""
        results = self.storage.search_by_content("DATABASE")

        assert len(results) == 1
        assert results[0].entry_id == "error-1"

    def test_search_by_content_in_tags(self):
        """Test text search in tags."""
        results = self.storage.search_by_content("performance")

        assert len(results) == 1
        assert results[0].entry_id == "warning-1"

    def test_search_by_content_partial_match(self):
        """Test partial text matching in content search."""
        results = self.storage.search_by_content("CPU")

        assert len(results) == 1
        assert results[0].entry_id == "warning-1"

    def test_search_by_content_no_matches(self):
        """Test content search with no matches."""
        results = self.storage.search_by_content("nonexistent_term")
        assert len(results) == 0


class TestKBStorageTagOperations:
    """Test tag-related operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()

        # Create entries with various tags
        entries = [
            KBEntry(
                entry_id="entry-1",
                content={"title": "Entry 1"},
                tags=["tag1", "tag2", "common"],
            ),
            KBEntry(
                entry_id="entry-2",
                content={"title": "Entry 2"},
                tags=["tag2", "tag3", "common"],
            ),
            KBEntry(
                entry_id="entry-3", content={"title": "Entry 3"}, tags=["tag1", "tag3"]
            ),
        ]

        for entry in entries:
            self.storage.store(entry)

    def test_get_all_tags(self):
        """Test retrieving all unique tags."""
        tags = self.storage.get_all_tags()

        expected_tags = ["common", "tag1", "tag2", "tag3"]
        assert tags == expected_tags  # Should be sorted

    def test_tag_index_consistency(self):
        """Test that tag index stays consistent."""
        # Check initial state
        assert len(self.storage._tag_index) == 4
        assert "entry-1" in self.storage._tag_index["tag1"]
        assert "entry-3" in self.storage._tag_index["tag1"]
        assert len(self.storage._tag_index["common"]) == 2

        # Delete an entry and check index cleanup
        self.storage.delete("entry-1")

        assert "entry-1" not in self.storage._tag_index["tag1"]
        assert "entry-3" in self.storage._tag_index["tag1"]  # Still there
        # tag2 should still exist because entry-2 has it
        assert "tag2" in self.storage._tag_index
        assert len(self.storage._tag_index["common"]) == 1

    def test_tag_update_index_management(self):
        """Test tag index management during updates."""
        # Update tags for entry-1
        new_tags = ["new_tag", "another_tag"]
        self.storage.update("entry-1", {"tags": new_tags})

        # Old tags should be removed from index
        assert "entry-1" not in self.storage._tag_index.get("tag1", set())
        assert "entry-1" not in self.storage._tag_index.get("tag2", set())
        assert "entry-1" not in self.storage._tag_index.get("common", set())

        # New tags should be in index
        assert "entry-1" in self.storage._tag_index["new_tag"]
        assert "entry-1" in self.storage._tag_index["another_tag"]


class TestKBStorageErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.storage = InMemoryKBStorage(logger=self.logger)

    def test_store_with_exception(self):
        """Test handling exceptions during store operations."""
        # Create a mock entry that will cause an exception
        bad_entry = Mock(spec=KBEntry)
        bad_entry.entry_id = "bad-entry"
        bad_entry.updated_at = None  # This will cause an exception

        # Mock the datetime to raise an exception
        with patch("polaris.common.kb_storage.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")

            result = self.storage.store(bad_entry)

            assert result is False
            self.logger.error.assert_called_once()

    def test_update_with_exception(self):
        """Test handling exceptions during update operations."""
        # Store a valid entry first
        entry = KBEntry(entry_id="test", content={"data": "test"})
        self.storage.store(entry)

        # Mock datetime to cause exception during update
        with patch("polaris.common.kb_storage.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")

            result = self.storage.update("test", {"content": {"new": "data"}})

            assert result is False
            self.logger.error.assert_called()

    def test_delete_with_exception(self):
        """Test handling exceptions during delete operations."""
        # Store entry
        entry = KBEntry(entry_id="test", content={"data": "test"}, tags=["tag1"])
        self.storage.store(entry)

        # Patch the logger to verify error logging
        with patch.object(self.storage, "_entries") as mock_entries:
            # Make the delete operation fail by raising an exception
            mock_entries.__delitem__.side_effect = Exception("Delete failed")
            mock_entries.__contains__.return_value = True  # Entry exists
            mock_entries.__getitem__.return_value = entry  # Return the entry

            result = self.storage.delete("test")

            assert result is False
            self.logger.error.assert_called()

    def test_stats_calculation(self):
        """Test statistics calculation with various data."""
        # Add some entries
        for i in range(5):
            entry = KBEntry(
                entry_id=f"entry-{i}",
                content={"data": f"test data {i}"},
                tags=[f"tag{i}", "common"],
            )
            self.storage.store(entry)

        stats = self.storage.get_stats()

        assert stats["total_entries"] == 5
        assert stats["total_tags"] == 6  # tag0-tag4 + common
        assert stats["memory_usage_estimate_kb"] > 0


class TestKBStorageIntegration:
    """Test integration scenarios and complex operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.storage = InMemoryKBStorage()

    def test_complete_lifecycle(self):
        """Test complete entry lifecycle."""
        # Create entry
        entry = KBEntry(
            entry_id="lifecycle-test",
            content={"title": "Test Entry", "status": "draft"},
            tags=["draft", "test"],
        )

        # Store
        assert self.storage.store(entry) is True
        assert self.storage.get("lifecycle-test") is not None

        # Update
        updates = {
            "content": {"status": "published"},
            "tags": ["published", "test"],
            "metadata": {"published_date": "2025-08-17"},
        }
        assert self.storage.update("lifecycle-test", updates) is True

        # Verify updates
        updated = self.storage.get("lifecycle-test")
        assert updated.content["status"] == "published"
        assert "published" in updated.tags
        assert "draft" not in updated.tags
        assert updated.metadata["published_date"] == "2025-08-17"

        # Search
        results = self.storage.search_by_filters({"status": "published"})
        assert len(results) == 1

        # Delete
        assert self.storage.delete("lifecycle-test") is True
        assert self.storage.get("lifecycle-test") is None

    def test_concurrent_operations_simulation(self):
        """Test simulated concurrent operations."""
        # Simulate multiple operations happening
        entries = []
        for i in range(10):
            entry = KBEntry(
                entry_id=f"concurrent-{i}",
                content={"index": i, "category": f"cat{i % 3}"},
                tags=[f"tag{i % 3}", "concurrent"],
            )
            entries.append(entry)
            self.storage.store(entry)

        # Verify all stored
        assert len(self.storage._entries) == 10

        # Update some entries
        for i in range(0, 10, 2):
            self.storage.update(f"concurrent-{i}", {"content": {"updated": True}})

        # Search operations
        cat0_results = self.storage.search_by_filters({"category": "cat0"})
        assert len(cat0_results) > 0

        tag_results = self.storage.search_by_content("concurrent")
        assert len(tag_results) == 10

        # Delete some entries
        for i in range(5, 10):
            self.storage.delete(f"concurrent-{i}")

        assert len(self.storage._entries) == 5

        # Verify tag index consistency
        all_tags = self.storage.get_all_tags()
        assert "concurrent" in all_tags


if __name__ == "__main__":
    pytest.main([__file__])
