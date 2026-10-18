from types import SimpleNamespace

import pytest

from src.services import llm_service as llm_service_module


@pytest.mark.asyncio
async def test_openai_compatible_provider_uses_configured_chat_completions(monkeypatch):
    request = {}
    client_options = {}

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        async def create_completion(self, **kwargs):
            request.update(kwargs)
            message = SimpleNamespace(content='{"ok": true}')
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            client_options.update(kwargs)
            self.client = FakeClient()

        def __getattr__(self, name):
            return getattr(self.client, name)

    monkeypatch.setattr(llm_service_module.settings, "llm_provider", "openai-compatible")
    monkeypatch.setattr(
        llm_service_module.settings,
        "openai_compatible_base_url",
        "https://llm.example.test/v1",
    )
    monkeypatch.setattr(llm_service_module.settings, "openai_compatible_api_key", "test-key")
    monkeypatch.setattr(llm_service_module.settings, "openai_compatible_model", "custom-model")
    monkeypatch.setattr(llm_service_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(llm_service_module.settings, "openai_api_key", "")
    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    service = llm_service_module.LLMService()
    result = await service.generate_json("system instructions", "user question")

    assert service.provider == "openai-compatible"
    assert client_options == {
        "api_key": "test-key",
        "base_url": "https://llm.example.test/v1",
    }
    assert request["model"] == "custom-model"
    assert request["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "user question"},
    ]
    assert request["response_format"] == {"type": "json_object"}
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_openai_compatible_provider_allows_blank_local_api_key(monkeypatch):
    client_options = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            client_options.update(kwargs)

    monkeypatch.setattr(llm_service_module.settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(llm_service_module.settings, "openai_compatible_api_key", "")
    monkeypatch.setattr(llm_service_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(llm_service_module.settings, "openai_api_key", "")
    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    service = llm_service_module.LLMService()

    assert service.provider == "openai-compatible"
    assert client_options["api_key"] == "not-needed"