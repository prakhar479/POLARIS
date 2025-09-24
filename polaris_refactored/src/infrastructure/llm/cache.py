"""
LLM Response Caching System

Provides in-memory caching for LLM responses and tool results
with TTL support and cache invalidation strategies.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .models import LLMRequest, LLMResponse, ToolResult
from ..caching import CacheEntry


@dataclass
class LLMCacheEntry:
    """Cache entry specifically for LLM responses."""
    request_hash: str
    response: LLMResponse
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    ttl_seconds: int = 300  # 5 minutes default
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class LLMCache:
    """In-memory cache for LLM responses and tool results."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300,  # 5 minutes
        cleanup_interval: int = 60  # 1 minute
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.logger = logging.getLogger(__name__)
        
        self._llm_cache: Dict[str, LLMCacheEntry] = {}
        self._tool_cache: Dict[str, CacheEntry[ToolResult]] = {}
        self._last_cleanup = datetime.now(timezone.utc)
        
        # Cache statistics
        self._stats = {
            "llm_hits": 0,
            "llm_misses": 0,
            "tool_hits": 0,
            "tool_misses": 0,
            "evictions": 0,
            "cleanups": 0
        }
    
    def _generate_request_hash(self, request: LLMRequest) -> str:
        """Generate a hash for an LLM request for caching."""
        # Create a normalized representation of the request
        cache_key_data = {
            "model": request.model_name,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in request.messages
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "tools": request.tools
        }
        
        # Convert to JSON string and hash
        json_str = json.dumps(cache_key_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _generate_tool_hash(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate a hash for tool execution parameters."""
        cache_key_data = {
            "tool_name": tool_name,
            "parameters": parameters
        }
        
        json_str = json.dumps(cache_key_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def get_llm_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get cached LLM response if available."""
        self._cleanup_if_needed()
        
        request_hash = self._generate_request_hash(request)
        entry = self._llm_cache.get(request_hash)
        
        if entry is None:
            self._stats["llm_misses"] += 1
            return None
        
        if entry.is_expired():
            del self._llm_cache[request_hash]
            self._stats["llm_misses"] += 1
            return None
        
        entry.touch()
        self._stats["llm_hits"] += 1
        self.logger.debug(f"LLM cache hit for request hash: {request_hash[:8]}...")
        
        return entry.response
    
    def cache_llm_response(
        self,
        request: LLMRequest,
        response: LLMResponse,
        ttl: Optional[int] = None
    ) -> None:
        """Cache an LLM response."""
        request_hash = self._generate_request_hash(request)
        
        entry = LLMCacheEntry(
            request_hash=request_hash,
            response=response,
            ttl_seconds=ttl or self.default_ttl,
            metadata={
                "model": request.model_name,
                "tokens": response.usage.get("total_tokens", 0)
            }
        )
        
        self._llm_cache[request_hash] = entry
        self.logger.debug(f"Cached LLM response for request hash: {request_hash[:8]}...")
        
        # Evict oldest entries if cache is full
        self._evict_if_needed()
    
    def get_tool_result(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[ToolResult]:
        """Get cached tool result if available."""
        self._cleanup_if_needed()
        
        tool_hash = self._generate_tool_hash(tool_name, parameters)
        entry = self._tool_cache.get(tool_hash)
        
        if entry is None:
            self._stats["tool_misses"] += 1
            return None
        
        if entry.is_expired():
            del self._tool_cache[tool_hash]
            self._stats["tool_misses"] += 1
            return None
        
        entry.touch()
        self._stats["tool_hits"] += 1
        self.logger.debug(f"Tool cache hit for {tool_name}: {tool_hash[:8]}...")
        
        return entry.value
    
    def cache_tool_result(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: ToolResult,
        ttl: Optional[int] = None
    ) -> None:
        """Cache a tool result."""
        # Only cache successful results
        if not result.success:
            return
        
        tool_hash = self._generate_tool_hash(tool_name, parameters)
        
        entry = CacheEntry(
            key=tool_hash,
            value=result,
            ttl_seconds=ttl or self.default_ttl,
            tags={tool_name}
        )
        
        self._tool_cache[tool_hash] = entry
        self.logger.debug(f"Cached tool result for {tool_name}: {tool_hash[:8]}...")
        
        # Evict oldest entries if cache is full
        self._evict_if_needed()
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        total_entries = len(self._llm_cache) + len(self._tool_cache)
        
        if total_entries <= self.max_size:
            return
        
        entries_to_evict = total_entries - self.max_size
        
        # Combine all entries with timestamps for eviction
        all_entries = []
        
        for key, entry in self._llm_cache.items():
            all_entries.append(("llm", key, entry.last_accessed))
        
        for key, entry in self._tool_cache.items():
            all_entries.append(("tool", key, entry.last_accessed))
        
        # Sort by last accessed time (oldest first)
        all_entries.sort(key=lambda x: x[2])
        
        # Evict oldest entries
        for i in range(entries_to_evict):
            cache_type, key, _ = all_entries[i]
            
            if cache_type == "llm":
                del self._llm_cache[key]
            else:
                del self._tool_cache[key]
            
            self._stats["evictions"] += 1
        
        self.logger.debug(f"Evicted {entries_to_evict} cache entries")
    
    def _cleanup_if_needed(self) -> None:
        """Clean up expired entries if cleanup interval has passed."""
        now = datetime.now(timezone.utc)
        
        if now - self._last_cleanup < timedelta(seconds=self.cleanup_interval):
            return
        
        self._cleanup_expired_entries()
        self._last_cleanup = now
    
    def _cleanup_expired_entries(self) -> None:
        """Remove all expired entries from cache."""
        expired_llm = [
            key for key, entry in self._llm_cache.items()
            if entry.is_expired()
        ]
        
        expired_tool = [
            key for key, entry in self._tool_cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_llm:
            del self._llm_cache[key]
        
        for key in expired_tool:
            del self._tool_cache[key]
        
        total_expired = len(expired_llm) + len(expired_tool)
        
        if total_expired > 0:
            self.logger.debug(f"Cleaned up {total_expired} expired cache entries")
            self._stats["cleanups"] += 1
    
    def invalidate_llm_cache(self, model_name: Optional[str] = None) -> int:
        """Invalidate LLM cache entries, optionally filtered by model."""
        if model_name is None:
            count = len(self._llm_cache)
            self._llm_cache.clear()
            return count
        
        keys_to_remove = [
            key for key, entry in self._llm_cache.items()
            if entry.metadata.get("model") == model_name
        ]
        
        for key in keys_to_remove:
            del self._llm_cache[key]
        
        return len(keys_to_remove)
    
    def invalidate_tool_cache(self, tool_name: Optional[str] = None) -> int:
        """Invalidate tool cache entries, optionally filtered by tool name."""
        if tool_name is None:
            count = len(self._tool_cache)
            self._tool_cache.clear()
            return count
        
        keys_to_remove = [
            key for key, entry in self._tool_cache.items()
            if tool_name in entry.tags
        ]
        
        for key in keys_to_remove:
            del self._tool_cache[key]
        
        return len(keys_to_remove)
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self._llm_cache.clear()
        self._tool_cache.clear()
        self.logger.info("Cleared all cache entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_llm_requests = self._stats["llm_hits"] + self._stats["llm_misses"]
        total_tool_requests = self._stats["tool_hits"] + self._stats["tool_misses"]
        
        return {
            "llm_cache": {
                "size": len(self._llm_cache),
                "hits": self._stats["llm_hits"],
                "misses": self._stats["llm_misses"],
                "hit_rate": self._stats["llm_hits"] / total_llm_requests if total_llm_requests > 0 else 0
            },
            "tool_cache": {
                "size": len(self._tool_cache),
                "hits": self._stats["tool_hits"],
                "misses": self._stats["tool_misses"],
                "hit_rate": self._stats["tool_hits"] / total_tool_requests if total_tool_requests > 0 else 0
            },
            "total_size": len(self._llm_cache) + len(self._tool_cache),
            "max_size": self.max_size,
            "evictions": self._stats["evictions"],
            "cleanups": self._stats["cleanups"]
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        llm_entries = []
        for key, entry in self._llm_cache.items():
            llm_entries.append({
                "hash": key[:8] + "...",
                "model": entry.metadata.get("model"),
                "tokens": entry.metadata.get("tokens"),
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "expired": entry.is_expired()
            })
        
        tool_entries = []
        for key, entry in self._tool_cache.items():
            tool_entries.append({
                "hash": key[:8] + "...",
                "tags": list(entry.tags),
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "access_count": entry.access_count,
                "ttl_seconds": entry.ttl_seconds,
                "expired": entry.is_expired()
            })
        
        return {
            "llm_entries": llm_entries,
            "tool_entries": tool_entries,
            "statistics": self.get_statistics()
        }