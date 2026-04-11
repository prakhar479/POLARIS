"""Tests for create_llm_client_from_config helper."""

from polaris.infrastructure import llm as llm_module


def test_create_llm_client_from_config_filters_non_client_keys(monkeypatch):
    captured = {}

    def fake_create(provider: str = "google", **kwargs):
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(llm_module, "create_llm_client", fake_create)

    _ = llm_module.create_llm_client_from_config(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "max_tokens": 123,
            "resilience": {"rps": 1.0},
            "temperature": 0.2,
            "steps_limit": 5,
        }
    )

    assert captured["provider"] == "openai"
    assert captured["kwargs"] == {
        "resilience": {"rps": 1.0},
        "model": "gpt-4o-mini",
        "max_tokens": 123,
    }


def test_create_llm_client_from_config_uses_default_provider_when_missing(monkeypatch):
    captured = {}

    def fake_create(provider: str = "google", **kwargs):
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(llm_module, "create_llm_client", fake_create)

    _ = llm_module.create_llm_client_from_config({"model": "foo"})

    assert captured["provider"] == "google"
    assert captured["kwargs"] == {"resilience": None, "model": "foo"}


def test_create_llm_client_from_config_drop_keys(monkeypatch):
    captured = {}

    def fake_create(provider: str = "google", **kwargs):
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(llm_module, "create_llm_client", fake_create)

    _ = llm_module.create_llm_client_from_config(
        {"provider": "openrouter", "model": "x", "site_url": "https://example.com"},
        drop_keys={"site_url"},
    )

    assert captured["provider"] == "openrouter"
    assert captured["kwargs"] == {"resilience": None, "model": "x"}
