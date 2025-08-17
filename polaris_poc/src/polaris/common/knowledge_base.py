"""
Query and response models for POLARIS Knowledge Base.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries supported by the knowledge base."""
    STRUCTURED = "structured"
    NATURAL_LANGUAGE = "natural_language"
    SEMANTIC = "semantic"


class KBQuery(BaseModel):
    """Knowledge base query model."""
    
    query_id: str = Field(default_factory=lambda: str(hash(datetime.now())))
    query_type: QueryType
    content: str = Field(..., description="The query content")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Structured filters")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Query context")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def __get_validators__(cls):
        yield from super().__get_validators__()
        yield cls.validate_content

    @staticmethod
    def validate_content(values):
        content = values.get('content')
        if not content or not content.strip():
            raise ValueError('content must not be empty')
        return values


class KBEntry(BaseModel):
    """Knowledge base entry model."""
    
    entry_id: str = Field(..., description="Unique entry identifier")
    content: Dict[str, Any] = Field(..., description="Entry content")
    tags: Optional[List[str]] = Field(default=None, description="Entry tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Entry metadata")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KBResponse(BaseModel):
    """Knowledge base query response model."""
    
    query_id: str
    success: bool
    results: List[KBEntry] = Field(default_factory=list)
    total_results: int = Field(default=0)
    message: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    processing_time_ms: Optional[float] = Field(default=None)