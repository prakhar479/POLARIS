"""
LLM Client for Polaris.

Provides abstraction over different LLM providers (Google Gemini, OpenAI, etc.)
"""

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from polaris.infrastructure.constants import DEFAULT_MAX_TOKENS


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class GoogleGeminiClient(LLMClient):
    """Google Gemini LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """Initialize Google Gemini client with API key and model."""
        # Support both GOOGLE_API_KEY and GEMINI_API_KEY for compatibility
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "Google API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using Google Gemini with error handling."""
        try:
            # import GenerationConfig
            from google.ai.generativelanguage_v1beta import GenerationConfig

            # Convert messages to Gemini format
            prompt_parts = []
            for msg in messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}\n")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}\n")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}\n")

            prompt = "\n".join(prompt_parts)
            gen_config: Dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if response_schema:
                gen_config["response_mime_type"] = "application/json"
                gen_config["response_schema"] = response_schema

            # generate_content() is synchronous — run in a thread-pool executor
            # so we don't block the asyncio event loop (P0 fix).
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(
                    prompt, generation_config=GenerationConfig(**gen_config)
                ),
            )

            if not response.text:
                raise ValueError("Empty response from Gemini API")

            finish_reason = (
                response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            )
            return LLMResponse(content=response.text, model=self.model, finish_reason=finish_reason)

        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize OpenAI client with API key and model."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        try:
            import openai

            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. " "Install with: pip install openai")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using OpenAI with error handling."""
        try:
            # Convert to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_schema:
                # Basic support if user passes pydantic model in future
                pass

            response = await self.client.chat.completions.create(**kwargs)

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
            )

        except ImportError:
            raise ImportError("openai package not installed. " "Install with: pip install openai")
        except Exception as e:
            # Wrap API errors with more context
            raise RuntimeError(f"OpenAI API error: {e}") from e


class GroqClient(LLMClient):
    """Groq LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b"):
        """Initialize Groq client with API key and model."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Set GROQ_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        try:
            from groq import Groq

            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("groq package not installed. " "Install with: pip install groq")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using Groq with error handling."""
        try:
            # Convert to Groq format (same as OpenAI)
            groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Run sync client in thread pool to make it async
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=1,
                    stream=False,  # Use non-streaming for simplicity
                    stop=None,
                ),
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from Groq API")

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
            )

        except ImportError:
            raise ImportError("groq package not installed. " "Install with: pip install groq")
        except Exception as e:
            # Wrap API errors with more context
            raise RuntimeError(f"Groq API error: {e}") from e


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
        self.max_retries = int(self.resilience.get("max_retries", 4))
        self.base_backoff_ms = int(self.resilience.get("base_backoff_ms", 200))
        self.max_backoff_ms = int(self.resilience.get("max_backoff_ms", 4000))

        self._semaphore = asyncio.Semaphore(concurrency)
        # token bucket
        self._capacity = max(1, burst)
        self._tokens = float(self._capacity)
        self._refill_rate = float(rps)
        self._last_refill = time.monotonic()

        # Build inner clients per API key (if provided)
        self._clients: List[LLMClient] = []
        keys_env_var = self.resilience.get("keys_env_var")
        keys: List[str] = []
        if keys_env_var and os.getenv(keys_env_var):
            keys = [k.strip() for k in os.getenv(keys_env_var, "").split(",") if k.strip()]
        else:
            if self.provider == "openai" and os.getenv("OPENAI_API_KEYS"):
                keys = [k.strip() for k in os.getenv("OPENAI_API_KEYS", "").split(",") if k.strip()]
            if self.provider in ("google", "gemini") and os.getenv("GEMINI_API_KEYS"):
                keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
            if self.provider == "groq" and os.getenv("GROQ_API_KEYS"):
                keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]

        if not keys:
            # Single client fallback using default env
            self._clients.append(self._create_provider_client())
        else:
            for key in keys:
                client = self._create_provider_client(api_key=key)
                self._clients.append(client)

        self._client_idx = 0

        # Setup logging to a dedicated file for LLM debugging
        log_dir = Path(self.resilience.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("polaris.llm")
        self._logger.setLevel(logging.INFO)
        log_path = log_dir / "llm_debug.log"
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith(str(log_path))
            for h in self._logger.handlers
        ):
            fh = logging.FileHandler(str(log_path))
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

    def _create_provider_client(self, api_key: Optional[str] = None) -> LLMClient:
        if self.provider == "openai":
            return OpenAIClient(
                api_key=api_key, model=str(self.model or self.inner_kwargs.get("model", "gpt-4"))
            )
        elif self.provider == "groq":
            return GroqClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "openai/gpt-oss-120b")),
            )
        # default to google
        return GoogleGeminiClient(
            api_key=api_key,
            model=str(self.model or self.inner_kwargs.get("model", "gemini-2.5-flash")),
        )

    def _current_client(self) -> LLMClient:
        return self._clients[self._client_idx % len(self._clients)]

    def _rotate_client(self) -> None:
        self._client_idx = (self._client_idx + 1) % len(self._clients)

    async def _acquire_token(self) -> None:
        """Acquire one token from the rate-limit bucket (thread-safe via asyncio.Lock)."""
        if not hasattr(self, "_token_lock"):
            # Lazily create the lock the first time (safe: single-threaded asyncio).
            self._token_lock: asyncio.Lock = asyncio.Lock()
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

    def _classify_retryable(self, err: Exception) -> Tuple[bool, bool, str]:
        msg = str(err).lower()
        is_rate = any(x in msg for x in ["rate limit", "429", "too many requests", "quota"])
        is_retryable = is_rate or any(
            x in msg for x in ["timeout", "timed out", "connection reset", "503", "502", "500"]
        )
        etype = "rate_limited" if is_rate else ("retryable" if is_retryable else "fatal")
        return is_retryable, is_rate, etype

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using resilient LLM client with retry logic."""
        await self._semaphore.acquire()
        try:
            await self._acquire_token()
            attempt = 0
            while True:
                attempt += 1
                start = time.monotonic()
                try:
                    resp = await self._current_client().generate(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_schema=response_schema,
                    )
                    latency_ms = int((time.monotonic() - start) * 1000)
                    self._logger.info(
                        "provider=%s model=%s status=success latency_ms=%d tokens=%s",
                        self.provider,
                        getattr(resp, "model", "unknown"),
                        latency_ms,
                        getattr(resp, "tokens_used", None),
                    )
                    # Also write response content for debugging
                    try:
                        preview = resp.content if len(resp.content) <= 2000 else resp.content[:2000]
                        self._logger.info("response_preview=%s", preview)
                    except Exception:
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
                    if not is_retryable or attempt > self.max_retries:
                        raise
                    backoff = min(self.max_backoff_ms, self.base_backoff_ms * (2 ** (attempt - 1)))
                    backoff = backoff * (0.5 + random.random())
                    await asyncio.sleep(backoff / 1000.0)
        finally:
            self._semaphore.release()

    def update_resilience(self, new_resilience: Dict[str, Any]) -> None:
        """Hot-update resilience parameters at runtime.

        Safe to call while requests are in-flight. New settings apply to subsequent calls.
        """
        self.resilience.update(new_resilience)
        # Update rate limiter
        if "burst" in new_resilience:
            try:
                new_cap = max(1, int(new_resilience["burst"]))
                self._capacity = new_cap
                self._tokens = min(self._tokens, float(self._capacity))
            except Exception:
                pass
        if "rps" in new_resilience:
            try:
                self._refill_rate = float(new_resilience["rps"])
            except Exception:
                pass
        # Update retries/backoff
        for k in ("max_retries", "base_backoff_ms", "max_backoff_ms"):
            if k in new_resilience:
                try:
                    setattr(self, k, int(new_resilience[k]))
                except Exception:
                    pass
        # Update concurrency by swapping semaphore
        if "concurrency" in new_resilience:
            try:
                new_conc = int(new_resilience["concurrency"])
                if new_conc > 0:
                    self._semaphore = asyncio.Semaphore(new_conc)
                # else ignore invalid
            except Exception:
                pass


def create_llm_client(provider: str = "google", **kwargs: Any) -> LLMClient:
    """Create LLM client for specified provider.

    Args:
        provider: 'google'/'gemini', 'openai', or 'groq'
        **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    provider_norm = provider.lower()
    if provider_norm == "gemini":
        provider_norm = "google"

    resilience: Optional[Dict[str, Any]] = kwargs.pop("resilience", None)
    model = kwargs.get("model")

    # If resilience is explicitly provided or env enables it, wrap with ResilientLLMClient
    enabled_env = os.getenv("LLM_RESILIENCE_ENABLED", "0").lower() in ("1", "true", "yes")
    has_multi_keys = (
        (provider_norm == "openai" and os.getenv("OPENAI_API_KEYS"))
        or (provider_norm == "google" and os.getenv("GEMINI_API_KEYS"))
        or (provider_norm == "groq" and os.getenv("GROQ_API_KEYS"))
    )

    if resilience or enabled_env or has_multi_keys:
        return ResilientLLMClient(
            provider=provider_norm, model=model, inner_kwargs=kwargs, resilience=resilience
        )

    if provider_norm == "google":
        return GoogleGeminiClient(**kwargs)
    elif provider_norm == "openai":
        return OpenAIClient(**kwargs)
    elif provider_norm == "groq":
        return GroqClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_norm}")
