"""
In-memory storage implementation for POLARIS Knowledge Base.
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from .query_models import KBEntry


class InMemoryKBStorage:
    """Simple in-memory storage for knowledge base entries."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._entries: Dict[str, KBEntry] = {}
        self._tag_index: Dict[str, set] = {}  # tag -> set of entry_ids

    def store(self, entry: KBEntry) -> bool:
        """Store an entry in the knowledge base."""
        try:
            # Update timestamp
            entry.updated_at = datetime.now(timezone.utc).isoformat()

            # Store entry
            self._entries[entry.entry_id] = entry

            # Update tag index
            if entry.tags:
                for tag in entry.tags:
                    if tag not in self._tag_index:
                        self._tag_index[tag] = set()
                    self._tag_index[tag].add(entry.entry_id)

            self.logger.debug(f"Stored entry {entry.entry_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store entry {entry.entry_id}: {e}")
            return False

    def get(self, entry_id: str) -> Optional[KBEntry]:
        """Retrieve entry by ID."""
        return self._entries.get(entry_id)

    def search_by_filters(self, filters: Dict[str, Any]) -> List[KBEntry]:
        """Search entries by structured filters with advanced metric support."""
        results = []

        for entry in self._entries.values():
            if self._matches_filters(entry, filters):
                results.append(entry)

        return results

    def _matches_filters(self, entry: KBEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches all filters with advanced operators."""
        for key, value in filters.items():
            if not self._matches_single_filter(entry, key, value):
                return False
        return True

    def _matches_single_filter(self, entry: KBEntry, key: str, value: Any) -> bool:
        """Check if entry matches a single filter with operator support."""
        # Handle special tags filter
        if key == "tags" and isinstance(value, list):
            if not entry.tags or not any(tag in entry.tags for tag in value):
                return False
            return True

        # Handle operator-based filters (e.g., "cpu_usage__gt": 80)
        if "__" in key:
            field, operator = key.rsplit("__", 1)
            field_value = self._get_field_value(entry, field)
            if field_value is None:
                return False
            return self._apply_operator(field_value, operator, value)

        # Handle exact match filters
        field_value = self._get_field_value(entry, key)
        if field_value is None:
            return False
        return field_value == value

    def _get_field_value(self, entry: KBEntry, field: str):
        """Get field value from entry, supporting nested fields."""
        # Support nested field access (e.g., "metrics.cpu.usage")
        if "." in field:
            parts = field.split(".")
            value = entry.content
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value

        # Check in content first
        if field in entry.content:
            return entry.content[field]

        # Check in metadata
        if entry.metadata and field in entry.metadata:
            return entry.metadata[field]

        return None

    def _apply_operator(
        self, field_value: Any, operator: str, compare_value: Any
    ) -> bool:
        """Apply comparison operator to field value."""
        try:
            # Convert values to numbers if possible for numeric comparisons
            if operator in ["gt", "gte", "lt", "lte"] and isinstance(
                field_value, (int, float, str)
            ):
                try:
                    field_num = (
                        float(field_value)
                        if isinstance(field_value, str)
                        else field_value
                    )
                    compare_num = (
                        float(compare_value)
                        if isinstance(compare_value, str)
                        else compare_value
                    )

                    if operator == "gt":
                        return field_num > compare_num
                    elif operator == "gte":
                        return field_num >= compare_num
                    elif operator == "lt":
                        return field_num < compare_num
                    elif operator == "lte":
                        return field_num <= compare_num
                except (ValueError, TypeError):
                    pass

            # String operations
            if operator == "contains" and isinstance(field_value, str):
                return str(compare_value).lower() in field_value.lower()
            elif operator == "startswith" and isinstance(field_value, str):
                return field_value.lower().startswith(str(compare_value).lower())
            elif operator == "endswith" and isinstance(field_value, str):
                return field_value.lower().endswith(str(compare_value).lower())
            elif operator == "regex" and isinstance(field_value, str):
                import re

                return bool(re.search(str(compare_value), field_value, re.IGNORECASE))

            # List operations
            elif operator == "in" and isinstance(compare_value, list):
                return field_value in compare_value
            elif operator == "not_in" and isinstance(compare_value, list):
                return field_value not in compare_value

            # Exact match with negation
            elif operator == "ne":  # not equal
                return field_value != compare_value

        except Exception:
            return False

        return False

    def search_by_metric_range(
        self, metric_name: str, min_value: float = None, max_value: float = None
    ) -> List[KBEntry]:
        """Convenient method for metric range queries."""
        filters = {}
        if min_value is not None:
            filters[f"{metric_name}__gte"] = min_value
        if max_value is not None:
            filters[f"{metric_name}__lte"] = max_value

        return self.search_by_filters(filters)

    def search_by_content(self, query: str) -> List[KBEntry]:
        """Simple text search in entry content."""
        results = []
        query_lower = query.lower()

        for entry in self._entries.values():
            # Search in content values
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                results.append(entry)
                continue

            # Search in tags
            if entry.tags:
                tag_str = " ".join(entry.tags).lower()
                if query_lower in tag_str:
                    results.append(entry)

        return results

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        return sorted(
            list(self._tag_index.keys())
        )  # Fixed: sorted for consistent results

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        import sys

        size = sys.getsizeof(self._entries) + sys.getsizeof(self._tag_index)
        for entry in self._entries.values():
            size += sys.getsizeof(entry)
            size += sys.getsizeof(entry.content)
            if entry.tags:
                size += sys.getsizeof(entry.tags)
            if entry.metadata:
                size += sys.getsizeof(entry.metadata)
        return {
            "total_entries": len(self._entries),
            "total_tags": len(self._tag_index),
            "memory_usage_estimate_kb": max(1, size // 1024),  # Ensure at least 1KB
        }

    def clear(self) -> None:
        """Clear all stored data."""
        self._entries.clear()
        self._tag_index.clear()
        self.logger.debug("Storage cleared")

    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entry."""
        if entry_id not in self._entries:
            return False

        try:
            entry = self._entries[entry_id]

            # Update fields
            if "content" in updates:
                entry.content.update(updates["content"])
            if "tags" in updates:
                # Remove old tags from index
                if entry.tags:
                    for tag in entry.tags:
                        if tag in self._tag_index:
                            self._tag_index[tag].discard(entry_id)
                            if not self._tag_index[tag]:
                                del self._tag_index[tag]

                # Set new tags
                entry.tags = updates["tags"]

                # Add new tags to index
                if entry.tags:
                    for tag in entry.tags:
                        if tag not in self._tag_index:
                            self._tag_index[tag] = set()
                        self._tag_index[tag].add(entry_id)

            if "metadata" in updates:
                if entry.metadata is None:
                    entry.metadata = {}
                entry.metadata.update(updates["metadata"])

            # Update timestamp
            entry.updated_at = datetime.now(timezone.utc).isoformat()

            return True
        except Exception as e:
            self.logger.error(f"Failed to update entry {entry_id}: {e}")
            return False

    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        if entry_id not in self._entries:
            return False

        try:
            entry = self._entries[entry_id]

            # Remove from tag index
            if entry.tags:
                for tag in entry.tags:
                    if tag in self._tag_index:
                        self._tag_index[tag].discard(entry_id)
                        if not self._tag_index[tag]:
                            del self._tag_index[tag]

            # Remove entry
            del self._entries[entry_id]

            self.logger.debug(f"Deleted entry {entry_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete entry {entry_id}: {e}")
            return False
