"""News Service — NewsAPI integration for structured news search."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()


class NewsService:
    """NewsAPI client for structured news article search."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    async def search_articles(
        self,
        query: str,
        sort_by: str = "relevancy",
        page_size: int = 10,
        language: str = "en",
    ) -> list[dict]:
        """
        Search for news articles related to a claim.

        Returns articles with title, url, source, published_date, and description.
        """
        if not settings.newsapi_key:
            logger.debug("news.no_key", msg="Skipping — no API key configured")
            return []

        try:
            response = await self.client.get(
                self.BASE_URL,
                params={
                    "q": query,
                    "sortBy": sort_by,
                    "pageSize": page_size,
                    "language": language,
                    "apiKey": settings.newsapi_key,
                },
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.warning("news.bad_status", status=data.get("status"))
                return []

            results = []
            for article in data.get("articles", []):
                url = article.get("url", "")
                results.append({
                    "title": article.get("title", ""),
                    "url": url,
                    "source_name": article.get("source", {}).get("name", ""),
                    "source_domain": urlparse(url).netloc if url else "",
                    "published_date": article.get("publishedAt", ""),
                    "description": article.get("description", ""),
                    "content": (article.get("content") or "")[:2000],
                })

            logger.debug("news.search", query=query[:80], results=len(results))
            return results

        except Exception as e:
            logger.error("news.error", error=redact_secrets(str(e)))
            return []

    async def close(self):
        if self._client:
            await self._client.aclose()


# Singleton
news_service = NewsService()
