"""Reusable resilience primitives for LLM provider wrappers."""

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from polaris.infrastructure.llm.contracts import LLMProviderError, get_provider_multi_key_env_var


def _coerce_int(value: object, default: int) -> int:
    """Best-effort conversion of config values to int."""
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def provider_multi_key_env_var(provider: str) -> Optional[str]:
    """Return the canonical multi-key environment variable for a provider."""
    return get_provider_multi_key_env_var(provider)


def has_provider_multi_keys(provider: str) -> bool:
    """Return True when multi-key environment variables are configured."""
    env_var = provider_multi_key_env_var(provider)
    return bool(env_var and os.getenv(env_var))


def resolve_provider_keys(provider: str, resilience: Optional[Dict[str, object]]) -> List[str]:
    """Resolve provider API keys from resilience config and environment."""
    cfg = resilience or {}
    keys_env_var = cfg.get("keys_env_var")

    if isinstance(keys_env_var, str) and os.getenv(keys_env_var):
        return [k.strip() for k in os.getenv(keys_env_var, "").split(",") if k.strip()]

    env_var = provider_multi_key_env_var(provider)
    if env_var and os.getenv(env_var):
        return [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]

    return []


@dataclass
class RetryPolicy:
    """Retry/backoff policy for provider calls."""

    max_retries: int = 4
    base_backoff_ms: int = 200
    max_backoff_ms: int = 4000

    @classmethod
    def from_resilience_config(cls, resilience: Optional[Dict[str, object]]) -> "RetryPolicy":
        """Build a retry policy from a resilience configuration mapping."""
        cfg = resilience or {}
        return cls(
            max_retries=_coerce_int(cfg.get("max_retries", 4), 4),
            base_backoff_ms=_coerce_int(cfg.get("base_backoff_ms", 200), 200),
            max_backoff_ms=_coerce_int(cfg.get("max_backoff_ms", 4000), 4000),
        )

    def should_stop(self, attempt: int, retryable: bool) -> bool:
        """Return True if retry loop should stop."""
        return (not retryable) or attempt > self.max_retries

    def backoff_seconds(self, attempt: int) -> float:
        """Compute exponential backoff with jitter in seconds."""
        raw_backoff_ms = self.base_backoff_ms * (2 ** (attempt - 1))
        jittered_ms = raw_backoff_ms * (0.5 + random.random())
        backoff_ms = min(float(self.max_backoff_ms), float(jittered_ms))
        return backoff_ms / 1000.0

    def update_from_resilience_config(self, resilience: Dict[str, object]) -> None:
        """Update retry policy parameters from config if present."""
        for key in ("max_retries", "base_backoff_ms", "max_backoff_ms"):
            if key not in resilience:
                continue
            try:
                setattr(self, key, _coerce_int(resilience[key], getattr(self, key)))
            except Exception:
                continue


class RetryClassifier:
    """Error classifier used to drive retry/key-rotation logic."""

    @staticmethod
    def classify(err: Exception) -> Tuple[bool, bool, str]:
        """Return (retryable, rate_limited, error_type)."""
        if isinstance(err, LLMProviderError):
            if err.blocked:
                return False, False, "blocked"
            error_type = (
                "rate_limited" if err.rate_limited else ("retryable" if err.retryable else "fatal")
            )
            return err.retryable, err.rate_limited, error_type

        msg = str(err).lower()
        cause_msg = str(err.__cause__).lower() if err.__cause__ else ""
        combined = msg + " " + cause_msg

        is_rate = any(x in combined for x in ["rate limit", "429", "too many requests", "quota"])
        is_retryable = is_rate or any(
            x in combined for x in ["timeout", "timed out", "connection reset", "503", "502", "500"]
        )

        is_blocked = any(x in combined for x in ["safety", "blocked response", "recitation"])
        if is_blocked:
            return False, False, "blocked"

        error_type = "rate_limited" if is_rate else ("retryable" if is_retryable else "fatal")
        return is_retryable, is_rate, error_type


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, rps: float, burst: int):
        """Initialize the bucket with the requested rate and burst size."""
        self._capacity = max(1, int(burst))
        self._tokens = float(self._capacity)
        self._refill_rate = float(rps)
        self._last_refill = time.monotonic()
        self._token_lock: asyncio.Lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire one token from the bucket."""
        while True:
            async with self._token_lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
                    self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            await asyncio.sleep(0.01)

    def update(self, *, rps: Optional[float] = None, burst: Optional[int] = None) -> None:
        """Update limiter parameters."""
        if burst is not None:
            self._capacity = max(1, int(burst))
            self._tokens = min(self._tokens, float(self._capacity))
        if rps is not None:
            self._refill_rate = float(rps)


def configure_resilience_logger(log_dir: str) -> logging.Logger:
    """Build/return configured resilience logger."""
    resolved_dir = Path(log_dir).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("polaris.llm")
    logger.setLevel(logging.INFO)

    log_path = str(resolved_dir / "llm_debug.log")
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", "") == log_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)

    return logger
