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
        """Search entries by structured filters."""
        results = []

        for entry in self._entries.values():
            match = True

            # Check filters
            for key, value in filters.items():
                if key == "tags" and isinstance(value, list):
                    if not entry.tags or not any(tag in entry.tags for tag in value):
                        match = False
                        break
                elif key in entry.content:
                    if entry.content[key] != value:
                        match = False
                        break
                elif entry.metadata and key in entry.metadata:
                    if entry.metadata[key] != value:
                        match = False
                        break
                else:
                    match = False
                    break

            if match:
                results.append(entry)

        return results

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
