"""LLM infrastructure."""

from typing import Any, Dict, Iterable, Optional

from polaris.infrastructure.llm.base import LLMClient, LLMMessage, LLMResponse
from polaris.infrastructure.llm.client import ResilientLLMClient, create_llm_client
from polaris.infrastructure.llm.contracts import (
    CANONICAL_LLM_PROVIDERS,
    LLMBlockedResponseError,
    LLMProviderCapabilities,
    LLMProviderError,
    LLMRateLimitError,
    get_provider_capabilities,
)
from polaris.infrastructure.llm.providers import (
    GoogleGeminiClient,
    GroqClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)

SUPPORTED_LLM_CLIENT_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "model",
        "max_tokens",
        "base_url",
        "site_url",
        "app_name",
        "generate_mode",
    }
)


def create_llm_client_from_config(
    llm_config: Optional[Dict[str, Any]],
    *,
    default_provider: str = "google",
    drop_keys: Optional[Iterable[str]] = None,
) -> LLMClient:
    """Create an LLM client from a mixed strategy/meta-learner config dict.

    This helper centralizes extraction of provider/resilience and avoids passing
    strategy-specific keys to provider client constructors.
    """
    cfg = dict(llm_config or {})

    provider_raw = cfg.pop("provider", default_provider)
    provider = str(provider_raw or default_provider)
    resilience = cfg.pop("resilience", None)

    for key in drop_keys or []:
        cfg.pop(key, None)

    llm_kwargs = {
        key: value
        for key, value in cfg.items()
        if key in SUPPORTED_LLM_CLIENT_CONFIG_KEYS and value is not None
    }
    return create_llm_client(provider, resilience=resilience, **llm_kwargs)


__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "GoogleGeminiClient",
    "OpenAIClient",
    "OpenRouterClient",
    "GroqClient",
    "OllamaClient",
    "ResilientLLMClient",
    "CANONICAL_LLM_PROVIDERS",
    "LLMProviderCapabilities",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMBlockedResponseError",
    "get_provider_capabilities",
    "create_llm_client",
    "create_llm_client_from_config",
    "SUPPORTED_LLM_CLIENT_CONFIG_KEYS",
]
