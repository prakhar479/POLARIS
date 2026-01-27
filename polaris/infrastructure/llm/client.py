"""
LLM Client for Polaris.

Provides abstraction over different LLM providers (Google Gemini, OpenAI, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""
    role: str  # 'system', 'user', or 'assistant'
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
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class GoogleGeminiClient(LLMClient):
    """Google Gemini LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Google API key not provided")

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
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate response using Google Gemini with error handling."""

        try:
            # Convert messages to Gemini format
            # Gemini uses a simpler format - just concatenate user messages
            prompt_parts = []
            for msg in messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}\n")
                elif msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}\n")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}\n")

            prompt = "\n".join(prompt_parts)

            # Generate response with timeout
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )

            if not response.text:
                raise ValueError("Empty response from Gemini API")

            return LLMResponse(
                content=response.text,
                model=self.model,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None
            )
            
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            # Wrap API errors with more context
            raise RuntimeError(f"Gemini API error: {e}") from e


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate response using OpenAI with error handling."""

        try:
            # Convert to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from OpenAI API")

            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason
            )
            
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        except Exception as e:
            # Wrap API errors with more context
            raise RuntimeError(f"OpenAI API error: {e}") from e


def create_llm_client(provider: str = "google", **kwargs) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        provider: 'google' or 'openai'
        **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    if provider.lower() == "google":
        return GoogleGeminiClient(**kwargs)
    elif provider.lower() == "openai":
        return OpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
