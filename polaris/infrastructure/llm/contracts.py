"""Provider contracts for Polaris LLM infrastructure.

This module centralizes provider constants, capability metadata, and typed
provider errors so runtime and configuration validation share the same source of
truth.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

CANONICAL_LLM_PROVIDER_ORDER: Tuple[str, ...] = (
    "google",
    "openai",
    "openrouter",
    "groq",
    "ollama",
)

CANONICAL_LLM_PROVIDERS: FrozenSet[str] = frozenset(CANONICAL_LLM_PROVIDER_ORDER)

LLM_PROVIDER_MODULE_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "google": ("google.generativeai", "google.ai.generativelanguage_v1beta"),
    "openai": ("openai",),
    "openrouter": ("openai",),
    "groq": ("groq",),
    # Ollama openai_compat mode depends on the openai SDK wrapper.
    "ollama": ("openai",),
}

LLM_PROVIDER_SINGLE_KEY_ENV_VARS: Dict[str, Tuple[str, ...]] = {
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    # Ollama is commonly local and typically does not require credentials.
    "ollama": (),
}

LLM_PROVIDER_MULTI_KEY_ENV_VARS: Dict[str, str] = {
    "google": "GEMINI_API_KEYS",
    "openai": "OPENAI_API_KEYS",
    "openrouter": "OPENROUTER_API_KEYS",
    "groq": "GROQ_API_KEYS",
}


@dataclass(frozen=True)
class LLMProviderCapabilities:
    """Feature flags for a provider runtime."""

    native_tools: bool = False
    structured_output: bool = False
    streaming: bool = False
    sync_backend: bool = False


_PROVIDER_CAPABILITIES: Dict[str, LLMProviderCapabilities] = {
    "google": LLMProviderCapabilities(
        native_tools=True,
        structured_output=True,
        streaming=False,
        sync_backend=True,
    ),
    "openai": LLMProviderCapabilities(
        native_tools=True,
        structured_output=False,
        streaming=False,
        sync_backend=False,
    ),
    "openrouter": LLMProviderCapabilities(
        native_tools=True,
        structured_output=False,
        streaming=False,
        sync_backend=False,
    ),
    "groq": LLMProviderCapabilities(
        native_tools=True,
        structured_output=False,
        streaming=False,
        sync_backend=True,
    ),
    "ollama": LLMProviderCapabilities(
        native_tools=False,
        structured_output=False,
        streaming=False,
        sync_backend=False,
    ),
}


def get_provider_required_modules(provider: str) -> Tuple[str, ...]:
    """Return importable module names required by a provider."""
    provider_norm = str(provider or "").strip().lower()
    return LLM_PROVIDER_MODULE_REQUIREMENTS.get(provider_norm, ())


def get_provider_single_key_env_vars(provider: str) -> Tuple[str, ...]:
    """Return provider credential env vars for single-key setups."""
    provider_norm = str(provider or "").strip().lower()
    return LLM_PROVIDER_SINGLE_KEY_ENV_VARS.get(provider_norm, ())


def get_provider_multi_key_env_var(provider: str) -> Optional[str]:
    """Return provider env var for multi-key rotation setups."""
    provider_norm = str(provider or "").strip().lower()
    return LLM_PROVIDER_MULTI_KEY_ENV_VARS.get(provider_norm)


def ordered_canonical_llm_providers() -> Sequence[str]:
    """Return canonical providers in stable display order."""
    return CANONICAL_LLM_PROVIDER_ORDER


def get_provider_capabilities(
    provider: str,
    *,
    ollama_mode: str = "openai_compat",
) -> LLMProviderCapabilities:
    """Return capability metadata for a provider/mode combination."""
    provider_norm = str(provider or "").strip().lower()

    if provider_norm == "ollama":
        mode_norm = str(ollama_mode or "openai_compat").strip().lower()
        if mode_norm in ("openai", "openai-compatible", "openai_compatible"):
            mode_norm = "openai_compat"
        if mode_norm == "openai_compat":
            return LLMProviderCapabilities(
                native_tools=True,
                structured_output=False,
                streaming=False,
                sync_backend=False,
            )
        return _PROVIDER_CAPABILITIES["ollama"]

    return _PROVIDER_CAPABILITIES.get(provider_norm, LLMProviderCapabilities())


class LLMProviderError(RuntimeError):
    """Typed provider error with retryability metadata."""

    retryable: bool = False
    rate_limited: bool = False
    blocked: bool = False
    code: str = "provider_error"

    def __init__(
        self,
        message: str,
        retryable: Optional[bool] = None,
        rate_limited: Optional[bool] = None,
        blocked: Optional[bool] = None,
        code: Optional[str] = None,
        /,
    ) -> None:
        """Initialize provider error metadata."""
        retryable_value = self.retryable if retryable is None else retryable
        rate_limited_value = self.rate_limited if rate_limited is None else rate_limited
        blocked_value = self.blocked if blocked is None else blocked
        code_value = self.code if code is None else code

        super().__init__(message, retryable_value, rate_limited_value, blocked_value, code_value)
        self.retryable = bool(retryable_value)
        self.rate_limited = bool(rate_limited_value)
        self.blocked = bool(blocked_value)
        self.code = str(code_value)

    def __str__(self) -> str:
        """Return the human-readable error message."""
        return str(self.args[0]) if self.args else ""


class LLMRateLimitError(LLMProviderError):
    """Provider error indicating a rate limit response."""

    retryable = True
    rate_limited = True
    code = "rate_limited"


class LLMBlockedResponseError(LLMProviderError):
    """Provider error indicating content/safety blocking."""

    blocked = True
    code = "blocked_response"
