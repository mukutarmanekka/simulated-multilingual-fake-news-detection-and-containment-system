"""Search Service — Web search abstraction for evidence retrieval.

Supports Brave Search API (primary) with content extraction from URLs.
"""

from __future__ import annotations

from urllib.parse import urlparse
from typing import Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..utils.http_retry import retry_on_transient_http
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()


class SearchService:
    """Web search abstraction for retrieving evidence documents."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        # Observability for live integration checks. No credentials here.
        self.last_provider: str | None = None
        self.last_error: str | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30, follow_redirects=True)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_on_transient_http,
        reraise=True,
    )
    async def search_brave(self, query: str, count: int = 10) -> list[dict]:
        """Search using Brave Search API."""
        if not settings.brave_search_api_key:
            logger.warning("search.brave.no_key")
            return []

        try:
            response = await self.client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={
                    "q": query,
                    "count": count,
                    "text_decorations": False,
                    "search_lang": "en",
                    "freshness": "py",  # past year
                },
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": settings.brave_search_api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published_date": item.get("page_age"),
                    "source_domain": urlparse(item.get("url", "")).netloc,
                })

            self.last_provider = "brave"
            self.last_error = None
            logger.debug("search.brave", query=query[:80], results=len(results))
            return results

        except Exception as e:
            self.last_error = type(e).__name__
            logger.error(
                "search.brave.error",
                query=query[:80],
                error=redact_secrets(str(e)),
            )
            raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_on_transient_http,
        reraise=True,
    )
    async def search_google_cse(self, query: str, count: int = 10) -> list[dict]:
        """Search using Google Custom Search Engine (fallback)."""
        if not settings.google_cse_api_key or not settings.google_cse_cx:
            return []

        try:
            response = await self.client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": settings.google_cse_api_key,
                    "cx": settings.google_cse_cx,
                    "q": query,
                    "num": min(count, 10),
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "description": item.get("snippet", ""),
                    "published_date": None,
                    "source_domain": urlparse(item.get("link", "")).netloc,
                })

            return results

        except Exception as e:
            self.last_error = type(e).__name__
            logger.error("search.google_cse.error", error=redact_secrets(str(e)))
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_on_transient_http,
        reraise=True,
    )
    async def search_tavily(self, query: str, count: int = 10) -> list[dict]:
        """Search using Tavily Search API."""
        if not settings.tavily_api_key:
            return []

        try:
            response = await self.client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_raw_content": False,
                    "max_results": count,
                },
                headers={
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("content", ""),
                    "published_date": item.get("published_date"),
                    "source_domain": urlparse(item.get("url", "")).netloc,
                })

            self.last_provider = "tavily"
            self.last_error = None
            logger.debug("search.tavily", query=query[:80], results=len(results))
            return results

        except Exception as e:
            self.last_error = type(e).__name__
            logger.error(
                "search.tavily.error",
                query=query[:80],
                error=redact_secrets(str(e)),
            )
            raise

    async def search(self, query: str, count: int = 10) -> list[dict]:
        """Search using the best available search provider."""
        self.last_provider = None
        self.last_error = None

        # Use Tavily if explicitly provided or if Brave key looks like a Tavily key
        effective_tavily_key = settings.tavily_api_key or (
            settings.brave_search_api_key if settings.brave_search_api_key and settings.brave_search_api_key.startswith("tvly-") else None
        )
        if effective_tavily_key:
            # Temporarily store the key for search_tavily to use
            try:
                object.__setattr__(settings, 'tavily_api_key', effective_tavily_key)
            except Exception:
                pass  # If frozen, the key is already set or we skip
            try:
                results = await self.search_tavily(query, count)
                if results:
                    self.last_provider = "tavily"
                    self.last_error = None
                    return results
            except Exception as e:
                self.last_error = type(e).__name__
                logger.warning(
                    "search.tavily_failed",
                    error=redact_secrets(str(e)),
                )

        # Try Brave first
        try:
            results = await self.search_brave(query, count)
            if results:
                self.last_provider = "brave"
                self.last_error = None
                return results
        except Exception as e:
            self.last_error = type(e).__name__
            logger.warning(
                "search.brave_failed",
                error=redact_secrets(str(e)),
            )

        # Then Google CSE
        try:
            results = await self.search_google_cse(query, count)
            if results:
                self.last_provider = "google_cse"
                self.last_error = None
                return results
        except Exception as e:
            self.last_error = type(e).__name__
            logger.warning(
                "search.google_cse_failed",
                error=redact_secrets(str(e)),
            )
            return []

        if self.last_error is None:
            self.last_error = "no_results"
        return []

    async def extract_content(self, url: str) -> Optional[str]:
        """Extract the main text content from a URL."""
        try:
            response = await self.client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; FakeNewsBot/1.0)",
                },
                timeout=8,
            )
            response.raise_for_status()
            html = response.text


            # Use trafilatura for content extraction
            try:
                import trafilatura
                text = trafilatura.extract(html, include_comments=False)
                if text:
                    return text[:5000]  # Limit to 5K chars
            except ImportError:
                pass

            # Fallback: basic extraction with BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove script/style
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            return text[:5000] if text else None

        except Exception as e:
            logger.debug(
                "search.extract.error",
                url=url[:80],
                error=redact_secrets(str(e)),
            )
            return None

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


# Singleton
search_service = SearchService()
