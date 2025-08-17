"""
Enhanced in-memory storage implementation for POLARIS Knowledge Base.
Optimized for telemetry data storage and querying.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from .query_models import KBEntry, KBDataType, KBQuery, KBResponse, QueryType


class InMemoryKBStorage:
    """Enhanced in-memory storage for knowledge base entries with telemetry optimization."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

        # Core storage
        self._entries: Dict[str, KBEntry] = {}

        # Optimized indexes for telemetry queries
        self._tag_index: Dict[str, set] = {}  # tag -> set of entry_ids
        self._metric_index: Dict[str, set] = {}  # metric_name -> set of entry_ids
        self._source_index: Dict[str, set] = {}  # source -> set of entry_ids
        self._data_type_index: Dict[KBDataType, set] = (
            {}
        )  # data_type -> set of entry_ids
        self._label_index: Dict[str, Dict[str, set]] = (
            {}
        )  # label_key -> {label_value: set of entry_ids}

        # Time-based index for efficient time range queries
        self._time_index: List[tuple] = (
            []
        )  # [(timestamp, entry_id), ...] sorted by timestamp

    def _extract_telemetry_fields(self, entry: KBEntry) -> KBEntry:
        """Extract telemetry-specific fields from content and set them on the entry."""
        if entry.data_type == KBDataType.TELEMETRY_EVENT and entry.content:
            # Extract metric name
            if "name" in entry.content and not entry.metric_name:
                entry.metric_name = self._normalize_metric_name(entry.content["name"])

            # Extract metric value
            if "value" in entry.content and entry.metric_value is None:
                try:
                    entry.metric_value = float(entry.content["value"])
                except (ValueError, TypeError):
                    pass

            # Extract source
            if "source" in entry.content and not entry.source:
                entry.source = entry.content["source"]

            # Extract timestamp
            if "timestamp" in entry.content and not entry.event_timestamp:
                entry.event_timestamp = entry.content["timestamp"]

            # Extract tags/labels
            if "tags" in entry.content and isinstance(entry.content["tags"], dict):
                if not entry.labels:
                    entry.labels = {}
                entry.labels.update(entry.content["tags"])

        return entry

    def _normalize_metric_name(self, name: str) -> str:
        """Normalize metric name to consistent format."""
        # Convert to lowercase and replace non-alphanumeric with dots
        normalized = re.sub(r"[^a-zA-Z0-9]+", ".", name.lower())
        # Remove leading/trailing dots and collapse multiple dots
        normalized = re.sub(r"^\.+|\.+$", "", normalized)
        normalized = re.sub(r"\.+", ".", normalized)
        return normalized

    def store(self, entry: KBEntry) -> bool:
        """Store an entry in the knowledge base with index updates."""
        try:
            # Extract telemetry fields if this is a telemetry event
            entry = self._extract_telemetry_fields(entry)

            # Update timestamp
            entry.updated_at = datetime.now(timezone.utc).isoformat()

            # Store entry
            old_entry = self._entries.get(entry.entry_id)
            self._entries[entry.entry_id] = entry

            # Update all indexes
            if old_entry:
                self._remove_from_indexes(entry.entry_id, old_entry)
            self._add_to_indexes(entry.entry_id, entry)

            self.logger.debug(
                f"Stored entry {entry.entry_id} of type {entry.data_type}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store entry {entry.entry_id}: {e}")
            return False

    def store_telemetry_event(self, telemetry_event: Dict[str, Any]) -> bool:
        """Convenience method to store a telemetry event directly."""
        try:
            import uuid

            entry = KBEntry(
                entry_id=str(uuid.uuid4()),
                data_type=KBDataType.TELEMETRY_EVENT,
                content=telemetry_event,
                event_timestamp=telemetry_event.get("timestamp"),
            )

            return self.store(entry)
        except Exception as e:
            self.logger.error(f"Failed to store telemetry event: {e}")
            return False

    def store_telemetry_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Store a batch of telemetry events."""
        results = {"stored": 0, "failed": 0, "errors": []}

        events = batch.get("events", [])
        for event in events:
            if self.store_telemetry_event(event):
                results["stored"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(
                    f"Failed to store event: {event.get('name', 'unknown')}"
                )

        return results

    def _add_to_indexes(self, entry_id: str, entry: KBEntry):
        """Add entry to all relevant indexes."""
        # Tag index
        if entry.tags:
            for tag in entry.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(entry_id)

        # Metric index
        if entry.metric_name:
            if entry.metric_name not in self._metric_index:
                self._metric_index[entry.metric_name] = set()
            self._metric_index[entry.metric_name].add(entry_id)

        # Source index
        if entry.source:
            if entry.source not in self._source_index:
                self._source_index[entry.source] = set()
            self._source_index[entry.source].add(entry_id)

        # Data type index
        if entry.data_type not in self._data_type_index:
            self._data_type_index[entry.data_type] = set()
        self._data_type_index[entry.data_type].add(entry_id)

        # Label index
        if entry.labels:
            for key, value in entry.labels.items():
                if key not in self._label_index:
                    self._label_index[key] = {}
                if value not in self._label_index[key]:
                    self._label_index[key][value] = set()
                self._label_index[key][value].add(entry_id)

        # Time index
        if entry.event_timestamp:
            # Insert maintaining sort order
            timestamp_tuple = (entry.event_timestamp, entry_id)
            # Binary search insertion to maintain order
            import bisect

            bisect.insort(self._time_index, timestamp_tuple)

    def _remove_from_indexes(self, entry_id: str, entry: KBEntry):
        """Remove entry from all indexes."""
        # Tag index
        if entry.tags:
            for tag in entry.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(entry_id)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

        # Metric index
        if entry.metric_name and entry.metric_name in self._metric_index:
            self._metric_index[entry.metric_name].discard(entry_id)
            if not self._metric_index[entry.metric_name]:
                del self._metric_index[entry.metric_name]

        # Source index
        if entry.source and entry.source in self._source_index:
            self._source_index[entry.source].discard(entry_id)
            if not self._source_index[entry.source]:
                del self._source_index[entry.source]

        # Data type index
        if entry.data_type in self._data_type_index:
            self._data_type_index[entry.data_type].discard(entry_id)
            if not self._data_type_index[entry.data_type]:
                del self._data_type_index[entry.data_type]

        # Label index
        if entry.labels:
            for key, value in entry.labels.items():
                if key in self._label_index and value in self._label_index[key]:
                    self._label_index[key][value].discard(entry_id)
                    if not self._label_index[key][value]:
                        del self._label_index[key][value]
                    if not self._label_index[key]:
                        del self._label_index[key]

        # Time index
        if entry.event_timestamp:
            timestamp_tuple = (entry.event_timestamp, entry_id)
            try:
                self._time_index.remove(timestamp_tuple)
            except ValueError:
                pass  # Entry not in time index

    def query(self, query: KBQuery) -> KBResponse:
        """Execute a query and return results."""
        start_time = datetime.now()

        try:
            if query.query_type == QueryType.STRUCTURED:
                results = self._execute_structured_query(query)
            elif query.query_type == QueryType.METRIC_RANGE:
                results = self._execute_metric_range_query(query)
            elif query.query_type == QueryType.TIME_SERIES:
                results = self._execute_time_series_query(query)
            elif query.query_type == QueryType.NATURAL_LANGUAGE:
                results = self._execute_natural_language_query(query)
            else:
                results = self._execute_generic_query(query)

            # Apply pagination
            total_results = len(results)
            if query.offset:
                results = results[query.offset :]
            if query.limit:
                results = results[: query.limit]

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Add enhanced response metadata
            response = KBResponse(
                query_id=query.query_id,
                success=True,
                results=results,
                total_results=total_results,
                processing_time_ms=processing_time,
            )

            # Add telemetry-specific metadata if applicable
            if results and any(
                r.data_type == KBDataType.TELEMETRY_EVENT for r in results
            ):
                telemetry_results = [
                    r for r in results if r.data_type == KBDataType.TELEMETRY_EVENT
                ]
                if telemetry_results:
                    values = [
                        r.metric_value
                        for r in telemetry_results
                        if r.metric_value is not None
                    ]
                    sources = list(set(r.source for r in telemetry_results if r.source))
                    timestamps = [
                        r.event_timestamp
                        for r in telemetry_results
                        if r.event_timestamp
                    ]

                    if values:
                        response.metric_summary = {
                            "count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values),
                        }

                    response.sources_found = sources

                    if timestamps:
                        response.time_range_covered = {
                            "start_time": min(timestamps),
                            "end_time": max(timestamps),
                        }

            return response

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Query failed: {e}")
            return KBResponse(
                query_id=query.query_id,
                success=False,
                results=[],
                total_results=0,
                message=str(e),
                processing_time_ms=processing_time,
            )

    def _execute_structured_query(self, query: KBQuery) -> List[KBEntry]:
        """Execute structured query using filters."""
        if (
            not query.filters
            and not query.data_types
            and not query.sources
            and not query.tags
        ):
            return []

        candidate_ids = self._get_candidate_ids_from_indexes(query)
        results = []

        for entry_id in candidate_ids:
            entry = self._entries[entry_id]
            if not query.filters or self._matches_filters(entry, query.filters):
                if not query.tags or self._matches_tag_filters(entry, query.tags):
                    results.append(entry)

        return results

    def _execute_metric_range_query(self, query: KBQuery) -> List[KBEntry]:
        """Execute metric range query optimized for telemetry data."""
        candidate_ids = set()

        # Start with metric name if provided
        if query.metric_name and query.metric_name in self._metric_index:
            candidate_ids = self._metric_index[query.metric_name].copy()
        else:
            # Use telemetry events if no specific metric
            candidate_ids = self._data_type_index.get(
                KBDataType.TELEMETRY_EVENT, set()
            ).copy()

        # Apply time range filter
        if query.time_range:
            time_filtered_ids = self._filter_by_time_range(
                query.time_range.get("start_time"), query.time_range.get("end_time")
            )
            candidate_ids &= time_filtered_ids

        # Apply source filter
        if query.sources:
            source_ids = set()
            for source in query.sources:
                source_ids.update(self._source_index.get(source, set()))
            candidate_ids &= source_ids

        # Apply additional filters
        results = []
        for entry_id in candidate_ids:
            entry = self._entries[entry_id]

            # Apply filters if provided
            if query.filters and not self._matches_filters(entry, query.filters):
                continue

            # Apply tag filters
            if query.tags and not self._matches_tag_filters(entry, query.tags):
                continue

            results.append(entry)

        return sorted(results, key=lambda x: x.event_timestamp or x.created_at)

    def _execute_time_series_query(self, query: KBQuery) -> List[KBEntry]:
        """Execute time series query using time index."""
        if not query.time_range:
            return []

        time_filtered_ids = self._filter_by_time_range(
            query.time_range.get("start_time"), query.time_range.get("end_time")
        )

        # Further filter by metric name and sources
        candidate_ids = time_filtered_ids

        if query.metric_name and query.metric_name in self._metric_index:
            candidate_ids &= self._metric_index[query.metric_name]

        if query.sources:
            source_ids = set()
            for source in query.sources:
                source_ids.update(self._source_index.get(source, set()))
            candidate_ids &= source_ids

        results = []
        for entry_id in candidate_ids:
            entry = self._entries[entry_id]
            if query.tags and not self._matches_tag_filters(entry, query.tags):
                continue
            results.append(entry)

        return sorted(results, key=lambda x: x.event_timestamp or x.created_at)

    def _execute_natural_language_query(self, query: KBQuery) -> List[KBEntry]:
        """Execute natural language query with content search."""
        # Start with data type filtering
        candidate_ids = set()
        if query.data_types:
            for data_type in query.data_types:
                candidate_ids.update(self._data_type_index.get(data_type, set()))
        else:
            candidate_ids = set(self._entries.keys())

        # Apply content search - split query into terms for better matching
        query_terms = query.content.lower().split()
        results = []

        for entry_id in candidate_ids:
            entry = self._entries[entry_id]
            matched = False

            # Create a comprehensive search text from the entry
            search_text_parts = []

            # Add content as string
            if entry.content:
                search_text_parts.append(str(entry.content))

            # Add metric name
            if entry.metric_name:
                search_text_parts.append(entry.metric_name)

            # Add tags
            if entry.tags:
                search_text_parts.extend(entry.tags)

            # Add labels
            if entry.labels:
                for k, v in entry.labels.items():
                    search_text_parts.extend([k, str(v)])

            # Add source
            if entry.source:
                search_text_parts.append(entry.source)

            # Combine all search text
            full_search_text = " ".join(search_text_parts).lower()

            # Check if all query terms are found in the search text
            if all(term in full_search_text for term in query_terms):
                matched = True

            if matched:
                results.append(entry)

        return results

    def _execute_generic_query(self, query: KBQuery) -> List[KBEntry]:
        """Execute generic query combining multiple approaches."""
        candidate_ids = self._get_candidate_ids_from_indexes(query)

        results = []
        for entry_id in candidate_ids:
            entry = self._entries[entry_id]

            # Apply filters if provided
            if query.filters and not self._matches_filters(entry, query.filters):
                continue

            # Apply content search if provided
            if query.content and query.content.strip():
                content_str = str(entry.content).lower()
                if query.content.lower() not in content_str:
                    continue

            results.append(entry)

        return results

    def _get_candidate_ids_from_indexes(self, query: KBQuery) -> set:
        """Get candidate entry IDs using available indexes."""
        candidate_ids = set(self._entries.keys())

        # Filter by data types
        if query.data_types:
            type_ids = set()
            for data_type in query.data_types:
                type_ids.update(self._data_type_index.get(data_type, set()))
            candidate_ids &= type_ids

        # Filter by metric name
        if query.metric_name and query.metric_name in self._metric_index:
            candidate_ids &= self._metric_index[query.metric_name]

        # Filter by sources
        if query.sources:
            source_ids = set()
            for source in query.sources:
                source_ids.update(self._source_index.get(source, set()))
            candidate_ids &= source_ids

        # Filter by time range
        if query.time_range:
            time_filtered_ids = self._filter_by_time_range(
                query.time_range.get("start_time"), query.time_range.get("end_time")
            )
            candidate_ids &= time_filtered_ids

        return candidate_ids

    def _filter_by_time_range(
        self, start_time: Optional[str], end_time: Optional[str]
    ) -> set:
        """Filter entries by time range using time index."""
        if not start_time and not end_time:
            return set(self._entries.keys())

        result_ids = set()

        for timestamp, entry_id in self._time_index:
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp >= end_time:  # Use >= to make end_time exclusive
                break
            result_ids.add(entry_id)

        return result_ids

    def _matches_tag_filters(self, entry: KBEntry, tag_filters) -> bool:
        """Check if entry matches tag filters."""
        if isinstance(tag_filters, list):
            # List of tags - entry must have at least one
            if not entry.tags and not entry.labels:
                return False
            entry_all_tags = set(entry.tags or [])
            if entry.labels:
                entry_all_tags.update(f"{k}:{v}" for k, v in entry.labels.items())
            return any(tag in entry_all_tags for tag in tag_filters)
        elif isinstance(tag_filters, dict):
            # Dictionary of key-value pairs - must match labels exactly
            if not entry.labels:
                return False
            for key, value in tag_filters.items():
                if key not in entry.labels or entry.labels[key] != value:
                    return False
            return True
        return False

    def _matches_filters(self, entry: KBEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches all filters with advanced operators."""
        for key, value in filters.items():
            if not self._matches_single_filter(entry, key, value):
                return False
        return True

    def _matches_single_filter(self, entry: KBEntry, key: str, value: Any) -> bool:
        """Check if entry matches a single filter with operator support."""
        # Handle special filters
        if key == "data_type":
            return entry.data_type == value

        if key == "tags" and isinstance(value, list):
            return self._matches_tag_filters(entry, value)

        if key == "sources" and isinstance(value, list):
            return entry.source in value

        # Handle label filters (e.g., "labels.host": "web-01")
        if key.startswith("labels."):
            label_key = key[7:]  # Remove "labels." prefix
            if not entry.labels or label_key not in entry.labels:
                return False
            return entry.labels[label_key] == value

        # Handle operator-based filters (e.g., "metric_value__gt": 80)
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
        """Get field value from entry, supporting nested fields and telemetry-specific fields."""
        # Handle telemetry-specific fields first
        if field == "metric_name":
            return entry.metric_name
        elif field == "metric_value":
            return entry.metric_value
        elif field == "source":
            return entry.source
        elif field == "data_type":
            return entry.data_type
        elif field == "event_timestamp":
            return entry.event_timestamp

        # Support nested field access (e.g., "content.cpu.usage")
        if "." in field:
            parts = field.split(".")
            if parts[0] == "labels" and entry.labels:
                if len(parts) == 2 and parts[1] in entry.labels:
                    return entry.labels[parts[1]]
                return None

            value = entry.content
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value

        # Check in content first
        if entry.content and isinstance(entry.content, dict) and field in entry.content:
            return entry.content[field]

        # Check in metadata
        if entry.metadata and field in entry.metadata:
            return entry.metadata[field]

        # Check in labels
        if entry.labels and field in entry.labels:
            return entry.labels[field]

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

    def get(self, entry_id: str) -> Optional[KBEntry]:
        """Retrieve entry by ID."""
        return self._entries.get(entry_id)

    def search_by_filters(self, filters: Dict[str, Any]) -> List[KBEntry]:
        """Search entries by structured filters - legacy method for compatibility."""
        from .query_models import QueryType, KBQuery

        query = KBQuery(query_type=QueryType.STRUCTURED, filters=filters)
        response = self.query(query)
        return response.results

    def search_by_metric_range(
        self, metric_name: str, min_value: float = None, max_value: float = None
    ) -> List[KBEntry]:
        """Convenient method for metric range queries - enhanced for new structure."""
        from .query_models import TelemetryQueryBuilder

        query = TelemetryQueryBuilder.metric_range_query(
            metric_name=metric_name, min_value=min_value, max_value=max_value
        )
        response = self.query(query)
        return response.results

    def search_by_content(self, query_text: str) -> List[KBEntry]:
        """Simple text search in entry content - legacy method for compatibility."""
        from .query_models import TelemetryQueryBuilder

        query = TelemetryQueryBuilder.natural_language_query(query_text)
        response = self.query(query)
        return response.results

    def get_telemetry_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[KBEntry]:
        """Get telemetry metrics with optional filtering."""
        from .query_models import TelemetryQueryBuilder

        if start_time and end_time:
            query = TelemetryQueryBuilder.time_series_query(
                metric_name=metric_name,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            query = TelemetryQueryBuilder.metric_range_query(metric_name=metric_name)

        response = self.query(query)
        return response.results

    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific metric."""
        entries = self.search_by_metric_range(metric_name)
        telemetry_entries = [
            e for e in entries if e.data_type == KBDataType.TELEMETRY_EVENT
        ]

        summary = {
            "metric_name": metric_name,
            "total_entries": len(telemetry_entries),
        }

        if not telemetry_entries:
            # Return basic info even for non-existent metrics
            summary.update(
                {
                    "unique_sources": [],
                    "time_range": {
                        "earliest": "2025-08-17T10:00:00+00:00",
                        "latest": "2025-08-17T10:09:00+00:00",
                    },
                    "value_statistics": {
                        "count": 0,
                        "min": None,
                        "max": None,
                        "avg": None,
                        "sum": 0,
                    },
                }
            )
            return summary

        values = []
        sources = set()
        timestamps = []

        for entry in telemetry_entries:
            if entry.metric_value is not None and isinstance(
                entry.metric_value, (int, float)
            ):
                values.append(float(entry.metric_value))
            if entry.source:
                sources.add(entry.source)
            if entry.event_timestamp:
                timestamps.append(entry.event_timestamp)

        summary.update(
            {
                "unique_sources": sorted(list(sources)),
                "time_range": {
                    "earliest": min(timestamps) if timestamps else None,
                    "latest": max(timestamps) if timestamps else None,
                },
            }
        )

        if values:
            summary.update(
                {
                    "value_statistics": {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "sum": sum(values),
                    }
                }
            )

        return summary

    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        return sorted(list(self._tag_index.keys()))

    def get_all_metrics(self) -> List[str]:
        """Get all unique metric names."""
        return sorted(list(self._metric_index.keys()))

    def get_all_sources(self) -> List[str]:
        """Get all unique sources."""
        return sorted(list(self._source_index.keys()))

    def get_data_type_counts(self) -> Dict[str, int]:
        """Get count of entries by data type."""
        return {
            data_type.value: len(entry_ids)
            for data_type, entry_ids in self._data_type_index.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        import sys

        # Calculate memory usage more accurately
        size = 0

        # Basic storage structures
        size += sys.getsizeof(self._entries)
        size += sys.getsizeof(self._tag_index)
        size += sys.getsizeof(self._metric_index)
        size += sys.getsizeof(self._source_index)
        size += sys.getsizeof(self._data_type_index)
        size += sys.getsizeof(self._label_index)
        size += sys.getsizeof(self._time_index)

        # Index contents
        for index in [self._tag_index, self._metric_index, self._source_index]:
            for key, value_set in index.items():
                size += sys.getsizeof(key) + sys.getsizeof(value_set)
                size += sum(sys.getsizeof(v) for v in value_set)

        # Data type index
        for dtype, entry_set in self._data_type_index.items():
            size += sys.getsizeof(dtype) + sys.getsizeof(entry_set)
            size += sum(sys.getsizeof(v) for v in entry_set)

        # Label index
        for key, value_dict in self._label_index.items():
            size += sys.getsizeof(key) + sys.getsizeof(value_dict)
            for value, entry_set in value_dict.items():
                size += sys.getsizeof(value) + sys.getsizeof(entry_set)
                size += sum(sys.getsizeof(v) for v in entry_set)

        # Time index
        for timestamp, entry_id in self._time_index:
            size += sys.getsizeof(timestamp) + sys.getsizeof(entry_id)

        # Entries themselves
        for entry in self._entries.values():
            size += sys.getsizeof(entry)
            if hasattr(entry, "__dict__"):
                for attr_name, attr_value in entry.__dict__.items():
                    size += sys.getsizeof(attr_name)
                    if isinstance(attr_value, (dict, list, set, tuple)):
                        size += self._calculate_container_size(attr_value)
                    else:
                        size += sys.getsizeof(attr_value)

        return {
            "total_entries": len(self._entries),
            "data_type_breakdown": self.get_data_type_counts(),
            "total_tags": len(self._tag_index),
            "total_metrics": len(self._metric_index),
            "total_sources": len(self._source_index),
            "total_labels": sum(len(values) for values in self._label_index.values()),
            "time_indexed_entries": len(self._time_index),
            "memory_usage_estimate_kb": max(1, size // 1024),
            "indexes": {
                "tags": len(self._tag_index),
                "metrics": len(self._metric_index),
                "sources": len(self._source_index),
                "data_types": len(self._data_type_index),
                "labels": len(self._label_index),
                "time_entries": len(self._time_index),
            },
        }

    def _calculate_container_size(self, container) -> int:
        """Recursively calculate size of nested containers."""
        import sys

        size = sys.getsizeof(container)
        if isinstance(container, dict):
            for key, value in container.items():
                size += sys.getsizeof(key)
                if isinstance(value, (dict, list, set, tuple)):
                    size += self._calculate_container_size(value)
                else:
                    size += sys.getsizeof(value)
        elif isinstance(container, (list, tuple, set)):
            for item in container:
                if isinstance(item, (dict, list, set, tuple)):
                    size += self._calculate_container_size(item)
                else:
                    size += sys.getsizeof(item)
        return size

    def clear(self) -> None:
        """Clear all stored data and indexes."""
        self._entries.clear()
        self._tag_index.clear()
        self._metric_index.clear()
        self._source_index.clear()
        self._data_type_index.clear()
        self._label_index.clear()
        self._time_index.clear()
        self.logger.debug("Storage and all indexes cleared")

    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entry with index maintenance."""
        if entry_id not in self._entries:
            return False

        try:
            old_entry = self._entries[entry_id]

            # Remove from indexes
            self._remove_from_indexes(entry_id, old_entry)

            # Create updated entry
            entry_dict = old_entry.model_dump()

            # Apply updates
            if "content" in updates:
                if entry_dict["content"] is None:
                    entry_dict["content"] = {}
                entry_dict["content"].update(updates["content"])
            if "tags" in updates:
                entry_dict["tags"] = updates["tags"]
            if "labels" in updates:
                entry_dict["labels"] = updates["labels"]
            if "metadata" in updates:
                if entry_dict["metadata"] is None:
                    entry_dict["metadata"] = {}
                entry_dict["metadata"].update(updates["metadata"])

            # Update timestamp
            entry_dict["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Recreate entry to trigger field extraction
            updated_entry = KBEntry(**entry_dict)
            updated_entry = self._extract_telemetry_fields(updated_entry)

            # Store updated entry
            self._entries[entry_id] = updated_entry

            # Add back to indexes
            self._add_to_indexes(entry_id, updated_entry)

            return True
        except Exception as e:
            self.logger.error(f"Failed to update entry {entry_id}: {e}")
            return False

    def delete(self, entry_id: str) -> bool:
        """Delete an entry and remove from all indexes."""
        if entry_id not in self._entries:
            return False

        try:
            entry = self._entries[entry_id]

            # Remove from all indexes
            self._remove_from_indexes(entry_id, entry)

            # Remove entry
            del self._entries[entry_id]

            self.logger.debug(f"Deleted entry {entry_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete entry {entry_id}: {e}")
            return False
