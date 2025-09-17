"""
In-memory implementation of the BaseKnowledgeBase interface with built-in
telemetry buffering and aggregation.
"""

import logging
import re
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from polaris.knowledge_base.base import BaseKnowledgeBase
from polaris.knowledge_base.models import (
    KBEntry,
    KBQuery,
    KBResponse,
    QueryType,
    KBDataType,
)


class InMemoryKnowledgeBase(BaseKnowledgeBase):
    """
    In-memory KB with internal handling for recent telemetry events. It uses
    a size-limited buffer to hold raw telemetry and automatically aggregates
    it into a permanent OBSERVATION entry when the buffer is full.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, telemetry_buffer_size: int = 50):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        # Permanent storage for processed/explicit knowledge
        self._entries: Dict[str, KBEntry] = {}

        # Indexes for fast querying of permanent entries
        self._data_type_index: Dict[KBDataType, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)

        # Internal buffer for raw telemetry events before aggregation
        self._telemetry_buffer_size = telemetry_buffer_size
        self._raw_telemetry_buffers: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self._telemetry_buffer_size)
        )

        self.logger.info(
            f"In-Memory Knowledge Base initialized with telemetry buffer size of {telemetry_buffer_size}."
        )

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extracts and normalizes keywords from a text string for indexing."""
        if not text:
            return set()
        # Simple tokenizer: lowercase, split by non-alphanumeric characters
        words = re.split(r"\W+", text.lower())
        # Remove empty strings and short words
        return {word for word in words if len(word) > 2}

    def _add_to_indexes(self, entry: KBEntry):
        """Adds a permanent entry's ID to the relevant indexes."""
        entry_id = entry.entry_id
        self._data_type_index[entry.data_type].add(entry_id)

        for tag in entry.tags:
            self._tag_index[tag.lower()].add(entry_id)

        # Index keywords from summary and content values
        keywords = self._extract_keywords(entry.summary or "")
        content_text = " ".join(
            str(v) for v in entry.content.values() if isinstance(v, (str, int, float))
        )
        keywords.update(self._extract_keywords(content_text))

        for keyword in keywords:
            self._keyword_index[keyword].add(entry_id)

    def _remove_from_indexes(self, entry: KBEntry):
        """Removes a permanent entry's ID from all indexes."""
        entry_id = entry.entry_id

        if entry_id in self._data_type_index.get(entry.data_type, set()):
            self._data_type_index[entry.data_type].remove(entry_id)

        for tag in entry.tags:
            tag_lower = tag.lower()
            if entry_id in self._tag_index.get(tag_lower, set()):
                self._tag_index[tag_lower].remove(entry_id)

        keywords_to_check = [k for k, v in self._keyword_index.items() if entry_id in v]
        for keyword in keywords_to_check:
            self._keyword_index[keyword].remove(entry_id)

    def _aggregate_and_store_buffer(self, buffer_key: Tuple[str, str], events: List[KBEntry]):
        """Calculates stats from a buffer and updates or creates an OBSERVATION entry with trend info."""
        metric_name, source = buffer_key
        values = [e.metric_value for e in events if isinstance(e.metric_value, (int, float))]
        if not values:
            return

        # Calculate current statistics
        count = len(values)
        current_avg = sum(values) / count
        min_val = min(values)
        max_val = max(values)

        # Look for existing observation entry
        observation_id = f"obs_{metric_name}_{source}".replace(".", "_").replace(" ", "_")
        existing_entry = self._entries.get(observation_id)

        if existing_entry:
            # Update existing entry with trend
            prev_avg = existing_entry.content.get("average_value", current_avg)
            total_updates = existing_entry.content.get("total_updates", 0) + 1

            # Simple trend calculation
            if abs(current_avg - prev_avg) < (prev_avg * 0.05):  # Within 5%
                trend = "stable"
            elif current_avg > prev_avg:
                trend = "increasing"
            else:
                trend = "decreasing"

            # Update the existing entry
            existing_entry.content.update(
                {
                    "statistic": "aggregation",
                    "count": count,
                    "average_value": current_avg,
                    "previous_average": prev_avg,
                    "min_value": min_val,
                    "max_value": max_val,
                    "trend": trend,
                    "total_updates": total_updates,
                    "last_update": datetime.now().isoformat(),
                    "time_window_start": events[0].timestamp,
                    "time_window_end": events[-1].timestamp,
                }
            )
            # Keep a simple numeric value for quick views
            existing_entry.metric_value = current_avg

            existing_entry.summary = (
                f"Updated metric '{metric_name}' from '{source}' (#{total_updates}). "
                f"Avg: {current_avg:.2f} (was {prev_avg:.2f}, trend: {trend}), "
                f"Min: {min_val}, Max: {max_val}."
            )

            self.logger.info(f"Updated observation: {existing_entry.summary}")
        else:
            # Create new observation entry
            summary = (
                f"First observation for '{metric_name}' from '{source}'. "
                f"Avg: {current_avg:.2f}, Min: {min_val}, Max: {max_val}."
            )

            new_entry = KBEntry(
                entry_id=observation_id,
                data_type=KBDataType.OBSERVATION,
                summary=summary,
                metric_name=metric_name,
                source=source,
                metric_value=current_avg,
                content={
                    "statistic": "aggregation",
                    "count": count,
                    "average_value": current_avg,
                    "min_value": min_val,
                    "max_value": max_val,
                    "trend": "baseline",
                    "total_updates": 1,
                    "last_update": datetime.now().isoformat(),
                    "time_window_start": events[0].timestamp,
                    "time_window_end": events[-1].timestamp,
                },
                tags=["aggregated", "observation", metric_name.lower(), source.lower()],
            )

            # Store the new observation entry
            self.store(new_entry)
            self.logger.info(f"Created new observation: {summary}")

    def _handle_raw_telemetry(self, entry: KBEntry) -> bool:
        """Handles buffering and aggregation of raw telemetry."""
        if not entry.metric_name or not entry.source:
            self.logger.warning(
                f"Raw telemetry entry {entry.entry_id} is missing metric_name or source. Discarding."
            )
            return False

        buffer_key = (entry.metric_name, entry.source)
        buffer = self._raw_telemetry_buffers[buffer_key]

        # If buffer is full, aggregate its current contents BEFORE adding the new one.
        if len(buffer) == self._telemetry_buffer_size:
            self._aggregate_and_store_buffer(buffer_key, list(buffer))
            # Clear the buffer after aggregation
            buffer.clear()

        buffer.append(entry)
        return True

    def store(self, entry: KBEntry) -> bool:
        """
        Stores an entry. If it's a RAW_TELEMETRY_EVENT, it's buffered for later
        aggregation. Otherwise, it's stored permanently and indexed.
        """

        # self.logger.info("Storing entry: %s", entry)

        if entry.data_type == KBDataType.RAW_TELEMETRY_EVENT:
            return self._handle_raw_telemetry(entry)

        # Original logic for all other permanent data types
        try:
            if entry.entry_id in self._entries:
                self.logger.info("Found existing entry, updating: %s", entry.entry_id)
                old_entry = self._entries[entry.entry_id]
                self.logger.info("Old entry: %s", old_entry)
                self._remove_from_indexes(old_entry)
                self.logger.info("Removed old entry from indexes.")

            self._entries[entry.entry_id] = entry
            self.logger.info("Added/Updated entry in main storage.")
            self._add_to_indexes(entry)
            self.logger.info("Added/Updated entry in indexes.")
            self.logger.info(f"Stored permanent entry {entry.entry_id} of type {entry.data_type}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store entry {entry.entry_id}: {e}", exc_info=True)
            return False

    def get(self, entry_id: str) -> Optional[KBEntry]:
        """Retrieves a single permanent entry by its unique ID."""
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Deletes a permanent entry from the knowledge base."""
        if entry_id not in self._entries:
            return False
        try:
            entry = self._entries.pop(entry_id)
            self._remove_from_indexes(entry)
            self.logger.debug(f"Deleted entry {entry_id}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete entry {entry_id}: {e}", exc_info=True)
            return False

    def query(self, query: KBQuery) -> KBResponse:
        """
        Executes a query against the knowledge base entries.
        """
        start_time = datetime.now()
        try:
            if query.query_type == QueryType.STRUCTURED:
                results = self._execute_structured_query(query)
            elif query.query_type == QueryType.NATURAL_LANGUAGE:
                results = self._execute_natural_language_query(query)
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")

            total_results = len(results)
            paginated_results = results[query.offset : query.offset + query.limit]
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                f"Query {query} returned {len(paginated_results)} of {total_results} total results in {processing_time:.2f} ms."
            )
            return KBResponse(
                query_id=query.query_id,
                success=True,
                results=paginated_results,
                total_results=total_results,
                processing_time_ms=processing_time,
            )
        except Exception as e:
            self.logger.error(f"Query {query.query_id} failed: {e}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return KBResponse(
                query_id=query.query_id,
                success=False,
                message=str(e),
                processing_time_ms=processing_time,
            )

    def _execute_structured_query(self, query: KBQuery) -> List[KBEntry]:
        candidate_ids = set(self._entries.keys())
        if query.data_types:
            type_ids = set()
            for dt in query.data_types:
                type_ids.update(self._data_type_index.get(dt, set()))
            candidate_ids &= type_ids
        if not query.filters:
            return [self._entries[eid] for eid in candidate_ids]
        results = []
        for entry_id in candidate_ids:
            entry = self._entries[entry_id]
            if self._matches_filters(entry, query.filters):
                results.append(entry)
        return results

    def _matches_filters(self, entry: KBEntry, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key == "tags":
                if not isinstance(value, list):
                    return False
                entry_tags = {t.lower() for t in entry.tags}
                filter_tags = {t.lower() for t in value}
                if not filter_tags.issubset(entry_tags):
                    return False
            elif key in entry.content:
                if entry.content[key] != value:
                    return False
            else:
                return False
        return True

    def _execute_natural_language_query(self, query: KBQuery) -> List[KBEntry]:
        if not query.query_text:
            return []
        query_keywords = self._extract_keywords(query.query_text)
        if not query_keywords:
            return []

        result_ids = self._keyword_index.get(query_keywords.pop(), set()).copy()
        for keyword in query_keywords:
            result_ids &= self._keyword_index.get(keyword, set())
            if not result_ids:
                break

        if query.data_types:
            type_ids = set()
            for dt in query.data_types:
                type_ids.update(self._data_type_index.get(dt, set()))
            result_ids &= type_ids

        return [self._entries[eid] for eid in result_ids]

    def clear(self) -> None:
        """Removes all permanent entries, indexes, and raw telemetry buffers."""
        self._entries.clear()
        self._data_type_index.clear()
        self._tag_index.clear()
        self._keyword_index.clear()
        self._raw_telemetry_buffers.clear()
        self.logger.info("Knowledge Base and all telemetry buffers cleared.")

    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the knowledge base store and buffers."""
        total_buffered_events = sum(len(buf) for buf in self._raw_telemetry_buffers.values())
        return {
            "total_permanent_entries": len(self._entries),
            "data_type_counts": {k.value: len(v) for k, v in self._data_type_index.items() if v},
            "unique_tags": len(self._tag_index),
            "indexed_keywords": len(self._keyword_index),
            "active_telemetry_buffers": len(self._raw_telemetry_buffers),
            "total_buffered_events": total_buffered_events,
        }
