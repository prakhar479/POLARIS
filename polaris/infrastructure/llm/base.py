"""Core LLM protocol and message/response models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from polaris.infrastructure.constants import DEFAULT_MAX_TOKENS
from polaris.infrastructure.llm.contracts import LLMProviderCapabilities


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
    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Normalized tool calls from native function-calling providers.

    Each entry is a dict with keys:
      - ``name`` (str): the function name called by the model
      - ``arguments`` (dict): the parsed JSON arguments

    None when the provider returned a plain text response, or when
    native tool calling is not active / not supported.
    """


class LLMClient(ABC):
    """Abstract LLM client interface."""

    def capabilities(self) -> LLMProviderCapabilities:
        """Return provider capabilities for this client instance."""
        return LLMProviderCapabilities()

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_schema: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            response_schema: Optional Pydantic schema for structured output.
            tools: Optional list of OpenAI-format function definitions to enable
                native tool calling. When provided, the model may respond with
                ``tool_calls`` instead of plain text.
            tool_choice: How the model selects tools. Common values: ``"auto"``
                (default), ``"none"``, or a specific function name.
        """
        pass

    async def generate_with_tools(
        self,
        messages: List[LLMMessage],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        """Generate a response using provider-native tool calling.

        This is an additive interface used by native tool-calling strategies.
        Provider clients should implement this and normalize tool calls into
        ``LLMResponse.tool_calls``.

        Raises:
            NotImplementedError: When a provider does not support native tools.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement native tool calling"
        )
