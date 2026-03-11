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

from polaris.infrastructure.constants import DEFAULT_GOOGLE_MODEL, DEFAULT_MAX_TOKENS


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

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GOOGLE_MODEL):
        """Initialize Google Gemini client with API key and model."""
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

    @staticmethod
    def _clean_schema_for_gemini(schema: Any) -> Any:
        """Convert Pydantic model schema to Gemini-compatible format.

        Gemini API has stricter schema requirements and doesn't accept:
        - "default" fields
        - "examples" fields
        - Other Pydantic-specific metadata

        This function extracts only the core schema structure that Gemini accepts.

        Args:
            schema: A Pydantic BaseModel class or already-extracted dict schema

        Returns:
            A cleaned schema compatible with Gemini API (can be any type)
        """
        if hasattr(schema, "model_json_schema"):
            schema_dict: Dict[str, Any] = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema
        else:
            raise TypeError(
                f"_clean_schema_for_gemini expects a Pydantic model class or a dict, "
                f"got {type(schema)!r}"
            )

        defs = schema_dict.get("$defs", {})
        if defs:
            schema_dict = GoogleGeminiClient._inline_refs(schema_dict, defs)

        return GoogleGeminiClient._recursively_clean_schema(schema_dict)

    @staticmethod
    def _inline_refs(obj: Any, defs: Dict[str, Any]) -> Any:
        """Replace $ref pointers with the referenced schema inline."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    return GoogleGeminiClient._inline_refs(defs[ref_name], defs)
                return obj  # Leave as-is if ref not found
            return {k: GoogleGeminiClient._inline_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [GoogleGeminiClient._inline_refs(item, defs) for item in obj]
        return obj

    @staticmethod
    def _recursively_clean_schema(obj: Any) -> Any:
        """Recursively convert Pydantic schema to Gemini-compatible format.

        Gemini API doesn't support:
        - "default" fields
        - "examples" fields
        - "title" fields
        - "anyOf" and "allOf" (needs flattening)
        - "$defs" and "$ref" (should be inlined before this step)

        This function converts the schema to be compatible with Gemini's limitations.

        Args:
            obj: Schema object (dict, list, or scalar)

        Returns:
            Gemini-compatible schema object
        """
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in obj.items():
                if key in {"default", "examples", "title", "$defs", "$ref", "additionalProperties"}:
                    continue

                if key == "anyOf":
                    # Handle anyOf by finding the most specific non-null type
                    cleaned.update(GoogleGeminiClient._flatten_anyof(value))
                elif key == "allOf":
                    # Handle allOf by merging schemas
                    merged = GoogleGeminiClient._merge_allof(value)
                    if merged:
                        cleaned.update(merged)
                else:
                    cleaned[key] = GoogleGeminiClient._recursively_clean_schema(value)
            return cleaned
        elif isinstance(obj, list):
            return [GoogleGeminiClient._recursively_clean_schema(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _flatten_anyof(anyof_list: List[Dict[str, Any]]) -> Any:
        """Flatten anyOf to a single schema that Gemini understands.

        For nullable fields (type + null), we use the non-null type.
        For multiple object types, we merge their properties.

        Returns:
            Schema object (can be dict or empty)
        """
        if not anyof_list:
            return {}

        # Filter out null types and find the most specific type
        non_null_options = [opt for opt in anyof_list if opt.get("type") != "null"]

        if not non_null_options:
            # All options are null, treat as optional
            return {}

        if len(non_null_options) == 1:
            # Single non-null option, use it directly
            return GoogleGeminiClient._recursively_clean_schema(non_null_options[0])

        merged: Dict[str, Any] = {"type": "object", "properties": {}}
        all_required: set = set()

        for option in non_null_options:
            cleaned_option = GoogleGeminiClient._recursively_clean_schema(option)
            if cleaned_option.get("type") == "object" and "properties" in cleaned_option:
                merged["properties"].update(cleaned_option["properties"])
                if "required" in cleaned_option:
                    all_required.update(cleaned_option["required"])
            else:
                # For non-object types, use the first one
                return GoogleGeminiClient._recursively_clean_schema(non_null_options[0])

        if all_required:
            merged["required"] = list(all_required)

        return merged

    @staticmethod
    def _merge_allof(allof_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge allOf schemas into a single schema."""
        if not allof_list:
            return {}

        merged: Dict[str, Any] = {"type": "object", "properties": {}}
        all_required: set = set()

        for schema in allof_list:
            cleaned_schema = GoogleGeminiClient._recursively_clean_schema(schema)
            if cleaned_schema.get("type") == "object":
                if "properties" in cleaned_schema:
                    merged["properties"].update(cleaned_schema["properties"])
                if "required" in cleaned_schema:
                    all_required.update(cleaned_schema["required"])
                # Copy other fields
                for key, value in cleaned_schema.items():
                    if key not in ("properties", "required", "type"):
                        merged[key] = value

        if all_required:
            merged["required"] = list(all_required)

        return merged

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using Google Gemini with error handling."""
        try:
            # Convert messages to Gemini format
            prompt_parts = []
            for msg in messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}\n")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}\n")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}\n")

            prompt = "".join(prompt_parts)

            from google.generativeai.types import generation_types

            gen_config: generation_types.GenerationConfigDict = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if response_schema:
                gen_config["response_mime_type"] = "application/json"
                gen_config["response_schema"] = self._clean_schema_for_gemini(response_schema)

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.generate_content(prompt, generation_config=gen_config),
            )

            if not response.candidates:
                raise ValueError(
                    "Gemini API returned no candidates (possible safety block or empty response)"
                )

            candidate = response.candidates[0]

            # Extract finish reason before touching .text
            finish_reason = None
            try:
                finish_reason = candidate.finish_reason.name
            except (AttributeError, IndexError):
                finish_reason = "UNKNOWN"

            # Check for safety/recitation blocks — these are not retryable
            non_stop_reasons = {"SAFETY", "RECITATION", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT"}
            if finish_reason in non_stop_reasons:
                raise ValueError(
                    f"Gemini API blocked response: finish_reason={finish_reason}. "
                    "This is not a transient error; do not retry."
                )

            # Now it is safe to access .text
            try:
                text = response.text
            except ValueError as ve:
                raise ValueError(f"Gemini API response text unavailable: {ve}") from ve

            if not text:
                raise ValueError("Empty response from Gemini API")

            return LLMResponse(content=text, model=self.model, finish_reason=finish_reason)

        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            if isinstance(e, (RuntimeError, ValueError)):
                raise
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
            raise ImportError("openai package not installed. Install with: pip install openai")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using OpenAI with error handling."""
        try:
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_schema is not None:
                logging.getLogger("polaris.llm").warning(
                    "OpenAIClient.generate: response_schema was provided but is not yet "
                    "implemented for OpenAI. The response will be unstructured text. "
                    "Pass a JSON-mode system prompt or use the structured-outputs API manually."
                )

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
            raise ImportError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            if isinstance(e, (RuntimeError, ValueError)):
                raise
            raise RuntimeError(f"OpenAI API error: {e}") from e


class GroqClient(LLMClient):
    """Groq LLM client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
    ):
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
            raise ImportError("groq package not installed. Install with: pip install groq")

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Generate response using Groq with error handling."""
        try:
            groq_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=groq_messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=1,
                    stream=False,
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
            raise ImportError("groq package not installed. Install with: pip install groq")
        except Exception as e:
            if isinstance(e, (RuntimeError, ValueError)):
                raise
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

        # Token bucket state
        self._capacity = max(1, burst)
        self._tokens = float(self._capacity)
        self._refill_rate = float(rps)
        self._last_refill = time.monotonic()
        self._token_lock: asyncio.Lock = asyncio.Lock()

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
            self._clients.append(self._create_provider_client())
        else:
            for key in keys:
                self._clients.append(self._create_provider_client(api_key=key))

        self._client_idx = 0
        log_dir = Path(self.resilience.get("log_dir", "./logs")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("polaris.llm")
        self._logger.setLevel(logging.INFO)
        log_path = str(log_dir / "llm_debug.log")
        if not any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
            for h in self._logger.handlers
        ):
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self._logger.addHandler(fh)

    def _create_provider_client(self, api_key: Optional[str] = None) -> LLMClient:
        if self.provider == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "gpt-4")),
            )
        elif self.provider == "groq":
            return GroqClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", "openai/gpt-oss-120b")),
            )
        elif self.provider in ("google", "gemini"):
            return GoogleGeminiClient(
                api_key=api_key,
                model=str(self.model or self.inner_kwargs.get("model", DEFAULT_GOOGLE_MODEL)),
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: '{self.provider}'. "
                "Supported values are: 'openai', 'groq', 'google', 'gemini'."
            )

    def _current_client(self) -> LLMClient:
        return self._clients[self._client_idx % len(self._clients)]

    def _rotate_client(self) -> None:
        self._client_idx = (self._client_idx + 1) % len(self._clients)

    async def _acquire_token(self) -> None:
        """Acquire one token from the rate-limit bucket."""
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
        cause_msg = str(err.__cause__).lower() if err.__cause__ else ""
        combined = msg + " " + cause_msg

        is_rate = any(x in combined for x in ["rate limit", "429", "too many requests", "quota"])
        is_retryable = is_rate or any(
            x in combined for x in ["timeout", "timed out", "connection reset", "503", "502", "500"]
        )
        # Safety-blocked responses should never be retried
        is_blocked = any(x in combined for x in ["safety", "blocked response", "recitation"])
        if is_blocked:
            is_retryable = False
            is_rate = False

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

                    raw_backoff = self.base_backoff_ms * (2 ** (attempt - 1))
                    jittered = raw_backoff * (0.5 + random.random())
                    backoff = min(self.max_backoff_ms, jittered)
                    await asyncio.sleep(backoff / 1000.0)
        finally:
            semaphore.release()

    def update_resilience(self, new_resilience: Dict[str, Any]) -> None:
        """Hot-update resilience parameters at runtime.

        Safe to call while requests are in-flight. New settings apply to
        subsequent calls.

        Note on concurrency: when the concurrency limit is changed, in-flight
        requests hold a reference to the old Semaphore and will release it
        correctly.  New requests will acquire the new
        Semaphore.  There is a brief window where both old and new semaphores
        are live; this is intentional and safe.
        """
        self.resilience.update(new_resilience)

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

        for k in ("max_retries", "base_backoff_ms", "max_backoff_ms"):
            if k in new_resilience:
                try:
                    setattr(self, k, int(new_resilience[k]))
                except Exception:
                    pass

        if "concurrency" in new_resilience:
            try:
                new_conc = int(new_resilience["concurrency"])
                if new_conc > 0:
                    self._semaphore = asyncio.Semaphore(new_conc)
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
