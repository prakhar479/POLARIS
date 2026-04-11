"""Tests for extracted LLM resilience primitives."""

import logging
from pathlib import Path

import pytest

from polaris.infrastructure.llm.resilience import (
    RetryClassifier,
    RetryPolicy,
    TokenBucketRateLimiter,
    configure_resilience_logger,
    has_provider_multi_keys,
    provider_multi_key_env_var,
    resolve_provider_keys,
)


def test_provider_multi_key_env_mapping():
    assert provider_multi_key_env_var("openai") == "OPENAI_API_KEYS"
    assert provider_multi_key_env_var("OPENROUTER") == "OPENROUTER_API_KEYS"
    assert provider_multi_key_env_var("unknown") is None


def test_has_provider_multi_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    assert has_provider_multi_keys("openai") is False

    monkeypatch.setenv("OPENAI_API_KEYS", "k1,k2")
    assert has_provider_multi_keys("openai") is True


def test_resolve_provider_keys_prefers_custom_env_var(monkeypatch):
    monkeypatch.setenv("CUSTOM_KEYS", "a, b")
    monkeypatch.setenv("OPENAI_API_KEYS", "fallback")

    keys = resolve_provider_keys("openai", {"keys_env_var": "CUSTOM_KEYS"})

    assert keys == ["a", "b"]


def test_resolve_provider_keys_falls_back_to_provider_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "g1,g2")

    keys = resolve_provider_keys("groq", {})

    assert keys == ["g1", "g2"]


def test_retry_policy_backoff_update_and_stop(monkeypatch):
    monkeypatch.setattr("polaris.infrastructure.llm.resilience.random.random", lambda: 0.0)

    policy = RetryPolicy(max_retries=2, base_backoff_ms=100, max_backoff_ms=1000)

    assert policy.backoff_seconds(1) == pytest.approx(0.05)
    assert policy.backoff_seconds(2) == pytest.approx(0.1)
    assert policy.should_stop(attempt=3, retryable=True) is True
    assert policy.should_stop(attempt=1, retryable=False) is True

    policy.update_from_resilience_config({"max_retries": "5", "base_backoff_ms": "250"})

    assert policy.max_retries == 5
    assert policy.base_backoff_ms == 250


def test_retry_classifier_string_fallbacks():
    is_retryable, is_rate, error_type = RetryClassifier.classify(RuntimeError("429 quota exceeded"))
    assert is_retryable is True
    assert is_rate is True
    assert error_type == "rate_limited"

    is_retryable, is_rate, error_type = RetryClassifier.classify(
        RuntimeError("Safety blocked response by provider")
    )
    assert is_retryable is False
    assert is_rate is False
    assert error_type == "blocked"


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter_update():
    limiter = TokenBucketRateLimiter(rps=100.0, burst=2)

    await limiter.acquire()
    await limiter.acquire()

    limiter.update(burst=3, rps=5.0)

    assert limiter._capacity == 3
    assert limiter._refill_rate == 5.0
    assert limiter._tokens <= 2.0


def test_configure_resilience_logger_is_idempotent(tmp_path):
    logger = configure_resilience_logger(str(tmp_path))
    configure_resilience_logger(str(tmp_path))

    expected_log_path = str(Path(tmp_path, "llm_debug.log").resolve())
    matching_handlers = [
        h
        for h in logger.handlers
        if isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == expected_log_path
    ]
    assert len(matching_handlers) == 1
