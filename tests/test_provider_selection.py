import types

import pytest

from polaris.core.polaris import Polaris
from polaris.infrastructure.config import PolarisConfig
from polaris.infrastructure.llm.client import ResilientLLMClient, create_llm_client


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
                "params": {"provider": "openai", "temperature": 0.1},
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
                "params": {"provider": "google", "steps_limit": 2},
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
                "params": {
                    "selection_mode": "first",
                    "strategies": [
                        {"type": "threshold", "priority": 0.9, "params": {}},
                        {
                            "type": "llm_reasoning",
                            "priority": 0.5,
                            "params": {"provider": "openai"},
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
    cfg = PolarisConfig.from_dict({"strategy": {"type": "agentic_llm", "params": {}}})

    _ = Polaris(config=cfg)
    assert captured.get("provider") == "google"


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


def test_openrouter_multi_keys_creates_resilient_client(monkeypatch):
    captured = []

    class DummyOpenRouterClient:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr("polaris.infrastructure.llm.client.OpenRouterClient", DummyOpenRouterClient)
    monkeypatch.delenv("LLM_RESILIENCE_ENABLED", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEYS", "key-1,key-2")
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)

    client = create_llm_client("openrouter")

    assert isinstance(client, ResilientLLMClient)
    assert len(client._clients) == 2
    assert [entry.get("api_key") for entry in captured] == ["key-1", "key-2"]
