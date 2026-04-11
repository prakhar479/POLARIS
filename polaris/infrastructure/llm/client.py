"""LLM factory and resilience wrapper for Polaris."""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from polaris.infrastructure.constants import DEFAULT_GOOGLE_MODEL, DEFAULT_MAX_TOKENS
from polaris.infrastructure.llm.base import LLMClient, LLMMessage, LLMResponse
from polaris.infrastructure.llm.contracts import CANONICAL_LLM_PROVIDERS, LLMProviderCapabilities
from polaris.infrastructure.llm.providers import (
    GoogleGeminiClient,
    GroqClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)
from polaris.infrastructure.llm.resilience import (
    RetryClassifier,
    RetryPolicy,
    TokenBucketRateLimiter,
    configure_resilience_logger,
    has_provider_multi_keys,
    resolve_provider_keys,
)


class ResilientLLMClient(LLMClient):
    """Resilience wrapper adding retries, rate limiting, and API key rotation."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        inner_kwargs: Optional[Dict[str, Any]] = None,
        resilience: Optional[Dict[str, Any]] = None,
    ):
        """Initialize resilient LLM client with provider and resilience settings."""
        self.provider = provider.lower()
        self.model = model
        self.inner_kwargs = inner_kwargs or {}
        self.resilience = resilience or {}

        rps = float(self.resilience.get("rps", 2.0))
        burst = int(self.resilience.get("burst", 4))
        concurrency = int(self.resilience.get("concurrency", burst))
        self._retry_policy = RetryPolicy.from_resilience_config(self.resilience)
        # Backward-compatible attribute mirrors
        self.max_retries = self._retry_policy.max_retries
        self.base_backoff_ms = self._retry_policy.base_backoff_ms
        self.max_backoff_ms = self._retry_policy.max_backoff_ms

        self._semaphore = asyncio.Semaphore(concurrency)
        self._rate_limiter = TokenBucketRateLimiter(rps=rps, burst=burst)
        self._sync_rate_limiter_state()

        # Build inner clients per API key (if provided)
        self._clients: List[LLMClient] = []
        keys = resolve_provider_keys(self.provider, self.resilience)

        if not keys:
            self._clients.append(self._create_provider_client())
        else:
            for key in keys:
                self._clients.append(self._create_provider_client(api_key=key))

        self._client_idx = 0
        self._logger = configure_resilience_logger(str(self.resilience.get("log_dir", "./logs")))

    def _sync_rate_limiter_state(self) -> None:
        """Keep legacy token-bucket attributes in sync with the limiter object."""
        # These mirrors preserve compatibility for any external diagnostics/tests.
        self._capacity = self._rate_limiter._capacity
        self._tokens = self._rate_limiter._tokens
        self._refill_rate = self._rate_limiter._refill_rate
        self._last_refill = self._rate_limiter._last_refill
        self._token_lock = self._rate_limiter._token_lock

    def _create_provider_client(self, api_key: Optional[str] = None) -> LLMClient:
        if self.provider == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "gpt-4")),
            )
        elif self.provider == "openrouter":
            return OpenRouterClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "openai/gpt-4o-mini")),
                base_url=(
                    str(self.inner_kwargs.get("base_url"))
                    if self.inner_kwargs.get("base_url")
                    else None
                ),
                site_url=(
                    str(self.inner_kwargs.get("site_url"))
                    if self.inner_kwargs.get("site_url")
                    else None
                ),
                app_name=(
                    str(self.inner_kwargs.get("app_name"))
                    if self.inner_kwargs.get("app_name")
                    else None
                ),
            )
        elif self.provider == "groq":
            return GroqClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "openai/gpt-oss-120b")),
            )
        elif self.provider == "google":
            return GoogleGeminiClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", DEFAULT_GOOGLE_MODEL)),
            )
        elif self.provider == "ollama":
            # Ollama typically doesn't require API keys. If provided, it's passed through.
            return OllamaClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "gpt-oss:20b")),
                base_url=(
                    str(self.inner_kwargs.get("base_url"))
                    if self.inner_kwargs.get("base_url")
                    else None
                ),
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: '{self.provider}'. "
                "Supported values are: 'google', 'openai', 'openrouter', 'groq', 'ollama'."
            )

    def _current_client(self) -> LLMClient:
        return self._clients[self._client_idx % len(self._clients)]

    def capabilities(self) -> LLMProviderCapabilities:
        """Return capabilities from the active inner client."""
        return self._current_client().capabilities()

    def _rotate_client(self) -> None:
        self._client_idx = (self._client_idx + 1) % len(self._clients)

    async def _acquire_token(self) -> None:
        """Acquire one token from the rate-limit bucket."""
        await self._rate_limiter.acquire()
        self._sync_rate_limiter_state()

    def _classify_retryable(self, err: Exception) -> Tuple[bool, bool, str]:
        return RetryClassifier.classify(err)

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """Generate response using resilient LLM client with retry logic."""
        semaphore = self._semaphore
        await semaphore.acquire()
        try:
            attempt = 0
            while True:
                attempt += 1
                await self._acquire_token()

                start = time.monotonic()
                try:
                    resp = await self._current_client().generate(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_schema=response_schema,
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                    latency_ms = int((time.monotonic() - start) * 1000)
                    self._logger.info(
                        "provider=%s model=%s status=success latency_ms=%d tokens=%s",
                        self.provider,
                        getattr(resp, "model", "unknown"),
                        latency_ms,
                        getattr(resp, "tokens_used", None),
                    )
                    try:
                        preview = resp.content if len(resp.content) <= 2000 else resp.content[:2000]
                        self._logger.info("response_preview=%s", preview)
                    except Exception:
                        # Silently ignore preview logging failures - not critical
                        pass
                    return resp

                except Exception as e:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    is_retryable, is_rate, etype = self._classify_retryable(e)
                    self._logger.info(
                        "provider=%s model=%s status=error error_type=%s latency_ms=%d error=%s attempt=%d",
                        self.provider,
                        self.model or "unknown",
                        etype,
                        latency_ms,
                        str(e).replace("\n", " ")[:512],
                        attempt,
                    )
                    if is_rate and len(self._clients) > 1:
                        self._rotate_client()
                        self._logger.info(
                            "provider=%s action=key_rotation new_index=%d",
                            self.provider,
                            self._client_idx,
                        )
                    if self._retry_policy.should_stop(attempt, is_retryable):
                        raise

                    await asyncio.sleep(self._retry_policy.backoff_seconds(attempt))
        finally:
            semaphore.release()

    async def generate_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        """Generate response using resilient retries for native tool calling."""
        semaphore = self._semaphore
        await semaphore.acquire()
        try:
            attempt = 0
            while True:
                attempt += 1
                await self._acquire_token()

                start = time.monotonic()
                try:
                    resp = await self._current_client().generate_with_tools(
                        messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    latency_ms = int((time.monotonic() - start) * 1000)
                    self._logger.info(
                        "provider=%s model=%s status=success latency_ms=%d tokens=%s",
                        self.provider,
                        getattr(resp, "model", "unknown"),
                        latency_ms,
                        getattr(resp, "tokens_used", None),
                    )
                    return resp

                except NotImplementedError:
                    # Explicit signal that this provider/mode does not support native tools.
                    raise
                except Exception as e:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    is_retryable, is_rate, etype = self._classify_retryable(e)
                    self._logger.info(
                        "provider=%s model=%s status=error error_type=%s latency_ms=%d error=%s attempt=%d",
                        self.provider,
                        self.model or "unknown",
                        etype,
                        latency_ms,
                        str(e).replace("\n", " ")[:512],
                        attempt,
                    )
                    if is_rate and len(self._clients) > 1:
                        self._rotate_client()
                        self._logger.info(
                            "provider=%s action=key_rotation new_index=%d",
                            self.provider,
                            self._client_idx,
                        )
                    if self._retry_policy.should_stop(attempt, is_retryable):
                        raise

                    await asyncio.sleep(self._retry_policy.backoff_seconds(attempt))
        finally:
            semaphore.release()

    def update_resilience(self, new_resilience: Dict[str, Any]) -> None:
        """Hot-update resilience parameters at runtime.

        Safe to call while requests are in-flight. New settings apply to subsequent
        calls.

        Note on concurrency: when the concurrency limit is changed, in-flight requests
        hold a reference to the old Semaphore and will release it correctly. New
        requests will acquire the new Semaphore. There is a brief window where both old
        and new semaphores are live; this is intentional and safe.
        """
        self.resilience.update(new_resilience)

        burst: Optional[int] = None
        if "burst" in new_resilience:
            try:
                burst = int(new_resilience["burst"])
            except Exception:
                # Silently ignore invalid burst value - keep existing setting
                pass

        rps: Optional[float] = None
        if "rps" in new_resilience:
            try:
                rps = float(new_resilience["rps"])
            except Exception:
                # Silently ignore invalid rps value - keep existing setting
                pass

        if burst is not None or rps is not None:
            try:
                self._rate_limiter.update(burst=burst, rps=rps)
                self._sync_rate_limiter_state()
            except Exception:
                pass

        self._retry_policy.update_from_resilience_config(new_resilience)
        self.max_retries = self._retry_policy.max_retries
        self.base_backoff_ms = self._retry_policy.base_backoff_ms
        self.max_backoff_ms = self._retry_policy.max_backoff_ms

        if "concurrency" in new_resilience:
            try:
                new_conc = int(new_resilience["concurrency"])
                if new_conc > 0:
                    self._semaphore = asyncio.Semaphore(new_conc)
            except Exception:
                # Silently ignore invalid concurrency value - keep existing setting
                pass


def create_llm_client(provider: str = "google", **kwargs: Any) -> LLMClient:
    """Create LLM client for specified provider.

    Args:
        provider: 'google', 'openai', 'openrouter', 'groq', or 'ollama'
            **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    provider_norm = str(provider or "").strip().lower()

    # Support aliases for Ollama
    if provider_norm in ("ollama-openai", "ollama_openai"):
        provider_norm = "ollama"

    if provider_norm not in CANONICAL_LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {provider_norm}. "
            "Supported values are: google, openai, openrouter, groq, ollama."
        )

    resilience: Optional[Dict[str, Any]] = kwargs.pop("resilience", None)
    model = kwargs.get("model")

    enabled_env = os.getenv("LLM_RESILIENCE_ENABLED", "0").lower() in ("1", "true", "yes")
    has_multi_keys = has_provider_multi_keys(provider_norm)

    if resilience or enabled_env or has_multi_keys:
        return ResilientLLMClient(
            provider=provider_norm, model=model, inner_kwargs=kwargs, resilience=resilience
        )

    if provider_norm == "google":
        return GoogleGeminiClient(**kwargs)
    elif provider_norm == "openai":
        return OpenAIClient(**kwargs)
    elif provider_norm == "openrouter":
        return OpenRouterClient(**kwargs)
    elif provider_norm == "groq":
        return GroqClient(**kwargs)
    elif provider_norm == "ollama":
        return OllamaClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider_norm}")
