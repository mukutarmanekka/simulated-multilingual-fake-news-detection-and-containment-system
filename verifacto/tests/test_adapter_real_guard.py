"""Phase 4: the mock-provider guard protects experiments from canned verdicts."""

import pytest

from verifacto.detector.base import DetectorUnavailableError, ensure_real_provider


def test_mock_provider_raises():
    with pytest.raises(DetectorUnavailableError):
        ensure_real_provider("mock")


def test_real_providers_pass():
    ensure_real_provider("gemini")
    ensure_real_provider("openai")
    ensure_real_provider("ollama")
