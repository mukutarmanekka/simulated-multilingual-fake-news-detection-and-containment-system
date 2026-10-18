"""LLM Service — Google Gemini and OpenAI-compatible API wrapper.

Provides structured JSON generation via LLMs with:
- Async-safe execution (sync Gemini client wrapped in asyncio.to_thread)
- Rate-limit-aware retry with exponential backoff
- JSON extraction from markdown-fenced responses
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from ..config import settings

logger = structlog.get_logger()


class LLMService:
    """Unified LLM interface supporting Gemini and OpenAI-compatible APIs."""

    def __init__(self):
        self._gemini_client = None
        self._openai_client = None
        self._openai_compatible_client = None
        self._deepseek_client = None
        self.provider = settings.llm_provider.strip().lower().replace("_", "-")
        self._init_client()

    def _init_client(self):
        """Initialize the configured LLM client with smart fallback."""
        # If Ollama configured, check if it's actually running
        if self.provider == "ollama":
            ollama_reachable = False
            try:
                import httpx
                r = httpx.get(f"{settings.ollama_url}/api/version", timeout=1.5)
                if r.status_code == 200:
                    ollama_reachable = True
            except Exception:
                ollama_reachable = False

            if not ollama_reachable:
                logger.warning(
                    "llm.ollama_unreachable",
                    url=settings.ollama_url,
                    msg="Ollama is not running. Checking for cloud API keys..."
                )
                if settings.gemini_api_key:
                    logger.info("llm.auto_switching", to="gemini")
                    self.provider = "gemini"
                elif settings.deepseek_api_key:
                    logger.info("llm.auto_switching", to="deepseek")
                    self.provider = "deepseek"
                elif settings.openai_api_key:
                    logger.info("llm.auto_switching", to="openai")
                    self.provider = "openai"
                else:
                    logger.info("llm.auto_switching", to="mock")
                    self.provider = "mock"
            else:
                logger.info("llm.init", provider="ollama", model=settings.ollama_model,
                            url=settings.ollama_url)
                return

        if self.provider == "mock":
            logger.info("llm.init", provider="mock", msg="Using mock LLM — no external calls")
            return

        if self.provider == "deepseek" and settings.deepseek_api_key:
            try:
                from openai import AsyncOpenAI
                self._deepseek_client = AsyncOpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url="https://api.deepseek.com",
                )
                logger.info("llm.init", provider="deepseek", model=settings.deepseek_model)
            except Exception as e:
                logger.error("llm.deepseek_init_failed", error=str(e))
        elif self.provider == "openai-compatible":
            try:
                from openai import AsyncOpenAI
                self._openai_compatible_client = AsyncOpenAI(
                    api_key=settings.openai_compatible_api_key or "not-needed",
                    base_url=settings.openai_compatible_base_url,
                )
                logger.info(
                    "llm.init",
                    provider="openai-compatible",
                    model=settings.openai_compatible_model,
                    base_url=settings.openai_compatible_base_url,
                )
            except Exception as e:
                logger.error("llm.openai_compatible_init_failed", error=str(e))
        elif self.provider == "gemini" and settings.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
                logger.info("llm.init", provider="gemini", model=settings.gemini_model)
            except Exception as e:
                logger.error("llm.gemini_init_failed", error=str(e))
        elif self.provider == "openai" and settings.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("llm.init", provider="openai", model=settings.openai_model)
            except Exception as e:
                logger.error("llm.openai_init_failed", error=str(e))

        # Fallback: only if the chosen API provider failed to init
        _any_client = (self._gemini_client or self._openai_client
                       or self._openai_compatible_client or self._deepseek_client)
        if not _any_client:
            if settings.gemini_api_key:
                try:
                    from google import genai
                    self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
                    self.provider = "gemini"
                    logger.info("llm.fallback", provider="gemini")
                except Exception:
                    pass
            if not self._gemini_client and settings.deepseek_api_key:
                try:
                    from openai import AsyncOpenAI
                    self._deepseek_client = AsyncOpenAI(
                        api_key=settings.deepseek_api_key,
                        base_url="https://api.deepseek.com",
                    )
                    self.provider = "deepseek"
                    logger.info("llm.fallback", provider="deepseek")
                except Exception:
                    pass
            if not self._gemini_client and not self._deepseek_client and settings.openai_api_key:
                try:
                    from openai import AsyncOpenAI
                    self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                    self.provider = "openai"
                    logger.info("llm.fallback", provider="openai")
                except Exception:
                    pass

        _any_client = (self._gemini_client or self._openai_client
                       or self._openai_compatible_client or self._deepseek_client)
        if not _any_client:
            logger.warning("llm.no_provider",
                           msg="No LLM provider configured — set GEMINI_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")
            self.provider = "mock"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict:
        """Generate a structured JSON response from the LLM."""
        start = time.monotonic()
        try:
            if self.provider == "gemini" and self._gemini_client:
                raw = await self._generate_gemini(system_prompt, user_prompt, temperature, max_tokens)
            elif self.provider == "deepseek" and self._deepseek_client:
                raw = await self._generate_deepseek(system_prompt, user_prompt, temperature, max_tokens)
            elif self.provider == "openai" and self._openai_client:
                raw = await self._generate_openai(system_prompt, user_prompt, temperature, max_tokens)
            elif self.provider == "openai-compatible" and self._openai_compatible_client:
                raw = await self._generate_openai_compatible(
                    system_prompt, user_prompt, temperature, max_tokens
                )
            elif self.provider == "ollama":
                raw = await self._generate_ollama(system_prompt, user_prompt, temperature, max_tokens)
            elif self.provider == "mock":
                raw = await self._generate_mock(system_prompt, user_prompt)
            else:
                # If chosen provider has no client, try mock fallback
                logger.warning("llm.provider_missing_client", provider=self.provider)
                raw = await self._generate_mock(system_prompt, user_prompt)

            result = self._parse_json(raw)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("llm.success", provider=self.provider, elapsed_ms=elapsed_ms)
            return result

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                logger.warning("llm.rate_limited", provider=self.provider,
                               msg="Rate limited — will retry with backoff")
            else:
                logger.error("llm.error", provider=self.provider, error=error_str)
            raise


    async def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate content using local Ollama instance."""
        import httpx
        url = f"{settings.ollama_url}/api/generate"
        payload = {
            "model": settings.ollama_model,
            "prompt": f"{system_prompt}\n\nUSER: {user_prompt}",
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def _generate_mock(self, system_prompt: str, user_prompt: str) -> str:
        """Return structured mock JSON for development/testing."""
        # Simple heuristic to determine which prompt is being used
        if "claim extraction" in system_prompt.lower():
            return json.dumps({
                "claims": [
                    {
                        "claim_id": "C1",
                        "text": "The input text contains factual information about the system.",
                        "entities": ["System"],
                        "dates": ["2026-04-21"],
                        "numbers": ["100%"],
                        "is_checkable": True
                    }
                ],
                "non_checkable_statements": ["This is a mock response."]
            })
        
        if "normalization" in system_prompt.lower() or "normalized" in user_prompt.lower():
            return json.dumps({
                "normalized_text": "The user requested a mock verification.",
                "entities": ["User"],
                "dates": ["Today"]
            })

        if "fact-checker" in system_prompt.lower() or "verify" in user_prompt.lower():
            return json.dumps({
                "subclaims": [
                    {
                        "text": "The system is currently in mock mode.",
                        "verdict": "SUPPORTED",
                        "reasoning": "Mock mode is active to bypass rate limits."
                    }
                ],
                "final_verdict": "SUPPORTED",
                "reasoning": [
                    "Verification was skipped and replaced with a high-fidelity mock response.",
                    "This allows for UI and database testing without external API calls."
                ]
            })

        return json.dumps({"message": "Mock response for unknown prompt type", "status": "ok"})

    async def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate content using Gemini API (wrapped in thread for async safety)."""
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )

        # Candidate models to try in order
        models_to_try = [settings.gemini_model, "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-flash-lite-latest"]
        seen = set()
        candidate_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        last_error = None
        for model_name in candidate_models:
            try:
                response = await asyncio.to_thread(
                    self._gemini_client.models.generate_content,
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                last_error = e
                err_text = str(e)
                if any(x in err_text for x in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND", "not available")):
                    logger.warning("llm.gemini_model_fallback", failed_model=model_name, error=err_text[:120])
                    continue
                raise

        if last_error:
            raise last_error
        return "{}"

    async def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate content using OpenAI API."""
        return await self._generate_openai_chat(
            self._openai_client,
            settings.openai_model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
        )

    async def _generate_deepseek(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate content using DeepSeek API (OpenAI-compatible)."""
        return await self._generate_openai_chat(
            self._deepseek_client,
            settings.deepseek_model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
        )

    async def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate content through an OpenAI-compatible chat completions API."""
        return await self._generate_openai_chat(
            self._openai_compatible_client,
            settings.openai_compatible_model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
        )

    @staticmethod
    async def _generate_openai_chat(
        client,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        msg = response.choices[0].message
        content = msg.content or ""
        if not content and getattr(msg, "reasoning_content", None):
            content = msg.reasoning_content
        return content

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
        if not text:
            return {}

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from markdown code fences
        patterns = [
            r'```json\s*\n(.*?)\n\s*```',
            r'```\s*\n(.*?)\n\s*```',
            r'\{.*\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    candidate = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(candidate)
                except (json.JSONDecodeError, IndexError):
                    continue

        logger.warning("llm.json_parse_failed", text_preview=text[:200])
        return {}


# Singleton instance
llm_service = LLMService()
