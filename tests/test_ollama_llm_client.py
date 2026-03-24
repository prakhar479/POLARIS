import pytest


def test_create_llm_client_ollama_provider():
    from polaris.infrastructure.llm import create_llm_client
    from polaris.infrastructure.llm.client import OllamaClient

    client = create_llm_client("ollama")
    assert isinstance(client, OllamaClient)


def test_create_llm_client_ollama_accepts_base_url_and_model():
    from polaris.infrastructure.llm import create_llm_client
    from polaris.infrastructure.llm.client import OllamaClient

    client = create_llm_client(
        "ollama",
        base_url="http://10.10.16.46:11435",
        model="gpt-oss:20b",
    )
    assert isinstance(client, OllamaClient)
    assert client.base_url == "http://10.10.16.46:11435"
    assert client.model == "gpt-oss:20b"

def test_ollama_native_mode_uses_api_generate(monkeypatch):
    from polaris.infrastructure.llm.client import LLMMessage, OllamaClient

    import asyncio

    captured = {}

    class DummyResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok", "eval_count": 12}

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return DummyResp()

    dummy_http_client = DummyAsyncClient()

    client = OllamaClient(
        base_url="http://10.10.16.46:11435",
        model="gpt-oss:20b",
        generate_mode="native",
        http_client=dummy_http_client,
    )

    resp = asyncio.run(
        client.generate(
            [
                LLMMessage(role="system", content="You are helpful"),
                LLMMessage(role="user", content="Hello"),
            ],
            temperature=0.3,
            max_tokens=123,
        )
    )

    assert resp.content == "ok"
    assert captured["url"] == "http://10.10.16.46:11435/api/generate"
    assert captured["json"]["model"] == "gpt-oss:20b"
    assert captured["json"]["stream"] is False
    assert "prompt" in captured["json"]
    assert captured["json"]["options"]["temperature"] == 0.3


@pytest.mark.parametrize("alias", ["ollama-openai", "ollama_openai"])
def test_create_llm_client_ollama_aliases(alias: str):
    from polaris.infrastructure.llm import create_llm_client
    from polaris.infrastructure.llm.client import OllamaClient

    client = create_llm_client(alias)
    assert isinstance(client, OllamaClient)
