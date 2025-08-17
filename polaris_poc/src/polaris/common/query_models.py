"""
Query and response models for POLARIS Knowledge Base.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, model_validator


class QueryType(str, Enum):
    """Types of queries supported by the knowledge base."""

    STRUCTURED = "structured"
    NATURAL_LANGUAGE = "natural_language"
    SEMANTIC = "semantic"


class KBQuery(BaseModel):
    """Knowledge base query model."""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_type: QueryType
    content: str = Field(
        default="", description="The query content"
    )  # Allow empty string
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured filters"
    )
    context: Optional[Dict[str, Any]] = Field(default=None, description="Query context")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @model_validator(mode="after")
    def validate_content_or_filters(self):
        """Validate that either content is provided or filters are provided."""
        if (not self.content or not self.content.strip()) and not self.filters:
            raise ValueError("content must not be empty when no filters are provided")
        return self


class KBEntry(BaseModel):
    """Knowledge base entry model."""

    entry_id: str = Field(..., description="Unique entry identifier")
    content: Dict[str, Any] = Field(..., description="Entry content")
    tags: Optional[List[str]] = Field(default=None, description="Entry tags")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Entry metadata"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class KBResponse(BaseModel):
    """Knowledge base query response model."""

    query_id: str
    success: bool
    results: List[KBEntry] = Field(default_factory=list)
    total_results: int = Field(default=0)
    message: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    processing_time_ms: Optional[float] = Field(default=None)
