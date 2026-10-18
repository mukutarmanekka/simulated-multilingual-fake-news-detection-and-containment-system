"""Phase 4: the mock-provider guard protects experiments from canned verdicts."""

from types import SimpleNamespace

import pytest

from verifacto.detector.base import DetectorUnavailableError, ensure_real_provider
from verifacto.detector.real_adapter import model_for_provider


def _settings():
    return SimpleNamespace(
        gemini_model="gemini-2.5-flash-lite",
        openai_model="gpt-4o",
        ollama_model="deepseek-r1:8b",
        deepseek_model="deepseek-chat",
        openai_compatible_model="local-model",
    )


def test_model_for_provider_known_providers():
    settings = _settings()
    assert model_for_provider("gemini", settings) == "gemini-2.5-flash-lite"
    assert model_for_provider("openai", settings) == "gpt-4o"
    assert model_for_provider("ollama", settings) == "deepseek-r1:8b"
    assert model_for_provider("deepseek", settings) == "deepseek-chat"
    assert model_for_provider("openai-compatible", settings) == "local-model"


def test_model_for_provider_unknown_falls_back_to_name():
    assert model_for_provider("mock", _settings()) == "mock"


def test_mock_provider_raises():
    with pytest.raises(DetectorUnavailableError):
        ensure_real_provider("mock")


def test_real_providers_pass():
    ensure_real_provider("gemini")
    ensure_real_provider("openai")
    ensure_real_provider("ollama")
