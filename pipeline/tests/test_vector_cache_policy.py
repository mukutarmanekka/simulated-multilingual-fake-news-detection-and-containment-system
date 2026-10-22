"""The detector's internal vector cache carries the verdict-policy version."""

import asyncio
from types import SimpleNamespace

from src.config import settings
from src.services.vector_service import VectorService


class _FakeClient:
    def __init__(self):
        self.payloads = []
        self.query_result = None

    def upsert(self, collection_name, points):
        self.payloads.extend(point.payload for point in points)

    def query_points(self, collection_name, query, limit, score_threshold):
        return self.query_result


def _service_with_client(client):
    service = VectorService()
    service._client = client
    service.encode = lambda text: [0.1, 0.2]
    return service


def test_store_claim_payload_includes_policy_version():
    client = _FakeClient()
    service = _service_with_client(client)

    asyncio.run(
        service.store_claim(
            claim_id="claim-1",
            claim_text="The Eiffel Tower is in Paris.",
            verdict="REAL",
            confidence=0.8,
        )
    )

    assert client.payloads
    assert client.payloads[0]["policy_version"] == settings.verdict_policy_version


def test_search_similar_returns_policy_version():
    client = _FakeClient()
    client.query_result = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="point-1",
                score=0.95,
                payload={
                    "claim_id": "claim-1",
                    "claim_text": "The Eiffel Tower is in Paris.",
                    "verdict": "REAL",
                    "confidence": 0.8,
                    "policy_version": settings.verdict_policy_version,
                },
            )
        ]
    )
    service = _service_with_client(client)

    matches = asyncio.run(
        service.search_similar("The Eiffel Tower is in Paris.", top_k=1)
    )

    assert matches[0]["policy_version"] == settings.verdict_policy_version
