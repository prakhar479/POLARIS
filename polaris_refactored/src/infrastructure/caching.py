"""
Multi-level caching strategies for POLARIS system.

This module implements L1 (in-memory) and L2 (persistent) caching with
invalidation policies and consistency management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, List, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from pathlib import Path
import logging


T = TypeVar('T')

class CacheLevel(Enum):
    """Cache level enumeration."""
    L1 = "L1"  # In-memory cache
    L2 = "L2"  # Persistent cache

class InvalidationPolicy(Enum):
    """Cache invalidation policy types."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    MANUAL = "manual"  # Manual invalidation only

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Determine which entries should be evicted."""
        pass

class TTLStrategy(CacheStrategy):
    """Time-to-live cache strategy."""
    
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Evict expired entries."""
        return [key for key, entry in entries.items() if entry.is_expired()]

class LRUStrategy(CacheStrategy):
    """Least recently used cache strategy."""
    
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Evict least recently used entries when over capacity."""
        if len(entries) <= max_size:
            return []
        
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].last_accessed
        )
        return [key for key, _ in sorted_entries[:len(entries) - max_size]]

class LFUStrategy(CacheStrategy):
    """Least frequently used cache strategy."""
    
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Evict least frequently used entries when over capacity."""
        if len(entries) <= max_size:
            return []
        
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed)
        )
        return [key for key, _ in sorted_entries[:len(entries) - max_size]]

@dataclass
class CacheConfiguration:
    """Cache configuration settings."""
    max_size: int = 1000
    default_ttl_seconds: Optional[int] = 3600  # 1 hour
    invalidation_policy: InvalidationPolicy = InvalidationPolicy.LRU
    enable_l2_cache: bool = True
    l2_cache_path: Optional[Path] = None
    cleanup_interval_seconds: int = 300  # 5 minutes
    enable_compression: bool = False

class L1Cache:
    """In-memory L1 cache implementation."""
    
    def __init__(self, config: CacheConfiguration):
        self.config = config
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._strategy = self._create_strategy()
        self._logger = logging.getLogger(__name__)
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _create_strategy(self) -> CacheStrategy:
        """Create cache strategy based on configuration."""
        if self.config.invalidation_policy == InvalidationPolicy.TTL:
            return TTLStrategy()
        elif self.config.invalidation_policy == InvalidationPolicy.LRU:
            return LRUStrategy()
        elif self.config.invalidation_policy == InvalidationPolicy.LFU:
            return LFUStrategy()
        else:
            return LRUStrategy()  # Default fallback
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                re