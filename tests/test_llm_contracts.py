"""Tests for provider contracts and typed LLM errors."""

from polaris.infrastructure.llm.client import ResilientLLMClient
from polaris.infrastructure.llm.contracts import (
    CANONICAL_LLM_PROVIDER_ORDER,
    CANONICAL_LLM_PROVIDERS,
    LLMBlockedResponseError,
    LLMRateLimitError,
    get_provider_capabilities,
    get_provider_multi_key_env_var,
    get_provider_required_modules,
    get_provider_single_key_env_vars,
    ordered_canonical_llm_providers,
)


def test_canonical_llm_providers_contains_expected_values():
    assert CANONICAL_LLM_PROVIDERS == frozenset(
        {"google", "openai", "openrouter", "groq", "ollama"}
    )


def test_ordered_canonical_provider_order_is_stable():
    assert tuple(ordered_canonical_llm_providers()) == CANONICAL_LLM_PROVIDER_ORDER
    assert CANONICAL_LLM_PROVIDER_ORDER == (
        "google",
        "openai",
        "openrouter",
        "groq",
        "ollama",
    )


def test_provider_metadata_helpers():
    assert get_provider_required_modules("openrouter") == ("openai",)
    assert get_provider_single_key_env_vars("google") == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    assert get_provider_multi_key_env_var("groq") == "GROQ_API_KEYS"
    assert get_provider_multi_key_env_var("ollama") is None


def test_get_provider_capabilities_openai():
    capabilities = get_provider_capabilities("openai")

    assert capabilities.native_tools is True
    assert capabilities.structured_output is False
    assert capabilities.sync_backend is False


def test_get_provider_capabilities_ollama_openai_mode():
    capabilities = get_provider_capabilities("ollama", ollama_mode="openai_compat")

    assert capabilities.native_tools is True
    assert capabilities.structured_output is False


def test_get_provider_capabilities_ollama_native_mode():
    capabilities = get_provider_capabilities("ollama", ollama_mode="native")

    assert capabilities.native_tools is False
    assert capabilities.structured_output is False


def test_resilient_classifier_handles_typed_rate_limit_error():
    client = object.__new__(ResilientLLMClient)

    is_retryable, is_rate, etype = client._classify_retryable(LLMRateLimitError("429"))

    assert is_retryable is True
    assert is_rate is True
    assert etype == "rate_limited"


def test_resilient_classifier_handles_typed_blocked_error():
    client = object.__new__(ResilientLLMClient)

    is_retryable, is_rate, etype = client._classify_retryable(LLMBlockedResponseError("blocked"))

    assert is_retryable is False
    assert is_rate is False
    assert etype == "blocked"
