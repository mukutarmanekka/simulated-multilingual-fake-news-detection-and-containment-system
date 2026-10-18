from types import SimpleNamespace
import pytest

from src.services import llm_service as llm_service_module


@pytest.mark.asyncio
async def test_deepseek_provider_uses_configured_chat_completions(monkeypatch):
    request = {}
    client_options = {}

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        async def create_completion(self, **kwargs):
            request.update(kwargs)
            message = SimpleNamespace(content='{"status": "ok", "provider": "deepseek"}')
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            client_options.update(kwargs)
            self.client = FakeClient()

        def __getattr__(self, name):
            return getattr(self.client, name)

    monkeypatch.setattr(llm_service_module.settings, "llm_provider", "deepseek")
    monkeypatch.setattr(llm_service_module.settings, "deepseek_api_key", "sk-deepseek-test-key")
    monkeypatch.setattr(llm_service_module.settings, "deepseek_model", "deepseek-chat")
    monkeypatch.setattr(llm_service_module.settings, "gemini_api_key", "")
    monkeypatch.setattr(llm_service_module.settings, "openai_api_key", "")
    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    service = llm_service_module.LLMService()
    result = await service.generate_json("system instructions", "user prompt")

    assert service.provider == "deepseek"
    assert client_options == {
        "api_key": "sk-deepseek-test-key",
        "base_url": "https://api.deepseek.com",
    }
    assert request["model"] == "deepseek-chat"
    assert request["messages"] == [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "user prompt"},
    ]
    assert request["response_format"] == {"type": "json_object"}
    assert result == {"status": "ok", "provider": "deepseek"}
