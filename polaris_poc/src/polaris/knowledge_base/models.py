"""
Core data models for the POLARIS Knowledge Base interface.

These models define the standard structures for entries, queries, and responses,
focusing on the needs of the adaptation and management system.
"""

import uuid
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class KBDataType(str, Enum):
    """Defines the types of knowledge stored."""

    RAW_TELEMETRY_EVENT = "raw_telemetry_event"  # <-- ADDED: For temporary storage
    ADAPTATION_DECISION = "adaptation_decision"
    SYSTEM_GOAL = "system_goal"
    LEARNED_PATTERN = "learned_pattern"
    OBSERVATION = "observation"  # e.g., summarized telemetry
    SYSTEM_INFO = "system_info"  # e.g., version, configuration
    GENERIC_FACT = "generic_fact"


class QueryType(str, Enum):
    """Supported query types."""

    STRUCTURED = "structured"
    NATURAL_LANGUAGE = "natural_language"


class KBEntry(BaseModel):
    """A generic entry in the knowledge base."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data_type: KBDataType
    summary: Optional[str] = Field(
        default=None, description="A brief, human-readable summary of the entry."
    )
    content: Dict[str, Any] = Field(
        default_factory=dict, description="The structured data content of the entry."
    )

    # --- ADDED: Dedicated fields for easier telemetry handling ---
    metric_name: Optional[str] = Field(default=None)
    metric_value: Optional[Union[float, int]] = Field(default=None)
    source: Optional[str] = Field(default=None)
    # -----------------------------------------------------------

    tags: List[str] = Field(
        default_factory=list, description="Keywords for searching and categorization."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of the event or when the knowledge was recorded.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional, non-indexed metadata."
    )


class KBQuery(BaseModel):
    """A query to be executed against the knowledge base."""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_type: QueryType
    query_text: Optional[str] = Field(
        default=None, description="The text for a natural language query."
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Key-value filters for a structured query."
    )
    data_types: Optional[List[KBDataType]] = Field(
        default=None, description="Filter results by one or more data types."
    )
    limit: int = Field(default=50, description="Maximum number of results to return.")
    offset: int = Field(default=0, description="Offset for pagination.")


class KBResponse(BaseModel):
    """The response from a knowledge base query."""

    query_id: str
    success: bool
    results: List[KBEntry] = Field(default_factory=list)
    total_results: int = 0
    message: Optional[str] = None
    processing_time_ms: Optional[float] = None
