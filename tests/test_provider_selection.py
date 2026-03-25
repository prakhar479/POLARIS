import types

import pytest

from polaris.core.polaris import Polaris
from polaris.infrastructure.config import PolarisConfig
from polaris.infrastructure.llm.client import create_llm_client


@pytest.mark.asyncio
async def test_llm_reasoning_provider_selection(monkeypatch):
    captured = {}

    def dummy_create_llm_client(provider: str = "google", **kwargs):
        captured["provider"] = provider
        # return a simple dummy with required interface
        dummy = types.SimpleNamespace()

        async def generate(messages, temperature=0.1, max_tokens=256):
            return types.SimpleNamespace(content="{}", model="dummy")

        dummy.generate = generate
        return dummy

    monkeypatch.setattr("polaris.infrastructure.llm.create_llm_client", dummy_create_llm_client)

    cfg = PolarisConfig.from_dict(
        {
            "strategy": {
                "type": "llm_reasoning",
                "llm_reasoning": {"provider": "openai", "temperature": 0.1},
            }
        }
    )

    polaris = Polaris(config=cfg)
    assert polaris.strategy is not None
    assert captured.get("provider") == "openai"


@pytest.mark.asyncio
async def test_agentic_llm_provider_selection(monkeypatch):
    captured = {}

    def dummy_create_llm_client(provider: str = "google", **kwargs):
        captured["provider"] = provider
        dummy = types.SimpleNamespace()

        async def generate(messages, temperature=0.1, max_tokens=256):
            return types.SimpleNamespace(content="{}", model="dummy")

        dummy.generate = generate
        return dummy

    monkeypatch.setattr("polaris.infrastructure.llm.create_llm_client", dummy_create_llm_client)

    cfg = PolarisConfig.from_dict(
        {
            "strategy": {
                "type": "agentic_llm",
                "agentic_llm": {"provider": "google", "steps_limit": 2},
            }
        }
    )

    polaris = Polaris(config=cfg)
    assert polaris.strategy is not None
    assert captured.get("provider") == "google"


@pytest.mark.asyncio
async def test_hybrid_sub_llm_provider_selection(monkeypatch):
    captured = {"providers": []}

    def dummy_create_llm_client(provider: str = "google", **kwargs):
        captured["providers"].append(provider)
        dummy = types.SimpleNamespace()

        async def generate(messages, temperature=0.1, max_tokens=256):
            return types.SimpleNamespace(content="{}", model="dummy")

        dummy.generate = generate
        return dummy

    monkeypatch.setattr("polaris.infrastructure.llm.create_llm_client", dummy_create_llm_client)

    cfg = PolarisConfig.from_dict(
        {
            "strategy": {
                "type": "hybrid",
                "hybrid": {
                    "selection_mode": "first",
                    "strategies": [
                        {"type": "threshold", "priority": 0.9},
                        {
                            "type": "llm_reasoning",
                            "priority": 0.5,
                            "llm_reasoning": {"provider": "openai"},
                        },
                    ],
                },
            }
        }
    )

    _ = Polaris(config=cfg)
    # Ensure an openai client was requested for the llm sub-strategy
    assert "openai" in captured["providers"]


@pytest.mark.asyncio
async def test_default_provider_is_google(monkeypatch):
    captured = {}

    def dummy_create_llm_client(provider: str = "google", **kwargs):
        captured["provider"] = provider
        dummy = types.SimpleNamespace()

        async def generate(messages, temperature=0.1, max_tokens=256):
            return types.SimpleNamespace(content="{}", model="dummy")

        dummy.generate = generate
        return dummy

    monkeypatch.setattr("polaris.infrastructure.llm.create_llm_client", dummy_create_llm_client)

    # No provider specified -> should default to google
    cfg = PolarisConfig.from_dict({"strategy": {"type": "agentic_llm", "agentic_llm": {}}})

    _ = Polaris(config=cfg)
    assert captured.get("provider") == "google"


def test_gemini_alias_maps_to_google_client(monkeypatch):
    captured = {}

    class DummyGoogleClient:
        def __init__(self, **kwargs):
            captured["called"] = True
            captured["kwargs"] = kwargs

    monkeypatch.setattr("polaris.infrastructure.llm.client.GoogleGeminiClient", DummyGoogleClient)
    monkeypatch.delenv("LLM_RESILIENCE_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)

    client = create_llm_client("gemini")
    assert isinstance(client, DummyGoogleClient)
    assert captured.get("called") is True


def test_openrouter_provider_creates_openrouter_client(monkeypatch):
    captured = {}

    class DummyOpenRouterClient:
        def __init__(self, **kwargs):
            captured["called"] = True
            captured["kwargs"] = kwargs

    monkeypatch.setattr("polaris.infrastructure.llm.client.OpenRouterClient", DummyOpenRouterClient)
    monkeypatch.delenv("LLM_RESILIENCE_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)

    client = create_llm_client("openrouter")
    assert isinstance(client, DummyOpenRouterClient)
    assert captured.get("called") is True


def test_openrouter_aliases_normalize(monkeypatch):
    captured = {}

    class DummyOpenRouterClient:
        def __init__(self, **kwargs):
            captured["called"] = True

    monkeypatch.setattr("polaris.infrastructure.llm.client.OpenRouterClient", DummyOpenRouterClient)
    monkeypatch.delenv("LLM_RESILIENCE_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)

    assert isinstance(create_llm_client("open-router"), DummyOpenRouterClient)
    assert isinstance(create_llm_client("open_router"), DummyOpenRouterClient)
