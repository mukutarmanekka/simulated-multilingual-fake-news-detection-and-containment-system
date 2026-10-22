"""Fact-Check Service — Google Fact Check Tools API integration.

Looks up existing fact-checks via the ClaimReview standard.
"""

from __future__ import annotations

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..utils.http_retry import RETRYABLE_STATUS_CODES, retry_on_transient_http
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()


class FactCheckService:
    """Google Fact Check Tools API client."""

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        # Observability for live integration checks. These never contain
        # credentials: only a short status/error category.
        self.last_error: str | None = None
        self.last_status: int | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_on_transient_http,
        reraise=True,
    )
    async def search_claims(
        self,
        query: str,
        language_code: str = "en",
        max_results: int = 10,
    ) -> list[dict]:
        """
        Search for existing fact-checks related to a claim.

        Returns a list of matching fact-checks with:
        - claim_text: The original claim reviewed
        - claimant: Who made the claim
        - rating: The fact-checker's rating
        - url: Link to the fact-check article
        - publisher: The fact-checking organization
        - review_date: When the review was published
        """
        self.last_error = None
        self.last_status = None

        if not settings.google_factcheck_api_key:
            self.last_error = "missing_api_key"
            logger.debug("factcheck.no_key", msg="Skipping — no API key configured")
            return []

        try:
            response = await self.client.get(
                self.BASE_URL,
                params={
                    "key": settings.google_factcheck_api_key,
                    "query": query,
                    "languageCode": language_code,
                    "pageSize": max_results,
                },
            )
            response.raise_for_status()
            self.last_status = response.status_code
            data = response.json()

            results = []
            for claim in data.get("claims", []):
                claim_text = claim.get("text", "")
                claimant = claim.get("claimant", "Unknown")

                for review in claim.get("claimReview", []):
                    results.append({
                        "claim_text": claim_text,
                        "claimant": claimant,
                        "rating": review.get("textualRating", ""),
                        "url": review.get("url", ""),
                        "publisher": review.get("publisher", {}).get("name", ""),
                        "review_date": review.get("reviewDate", ""),
                        "title": review.get("title", ""),
                        "language": review.get("languageCode", "en"),
                    })

            logger.debug("factcheck.search", query=query[:80], results=len(results))
            return results

        except httpx.HTTPStatusError as e:
            self.last_status = e.response.status_code
            if e.response.status_code == 429:
                self.last_error = "rate_limited"
                logger.warning("factcheck.rate_limited")
            else:
                self.last_error = f"http_{e.response.status_code}"
                logger.error("factcheck.error", status=e.response.status_code)

            if e.response.status_code in RETRYABLE_STATUS_CODES:
                # Let tenacity retry transient 429/5xx responses.
                raise
            return []
        except (httpx.TimeoutException, httpx.TransportError) as e:
            self.last_error = type(e).__name__
            logger.error(
                "factcheck.transport_error",
                error=redact_secrets(str(e)),
            )
            raise
        except Exception as e:
            self.last_error = type(e).__name__
            logger.error("factcheck.error", error=redact_secrets(str(e)))
            return []

    async def close(self):
        if self._client:
            await self._client.aclose()


# Singleton
factcheck_service = FactCheckService()
