

from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()


class VectorService:

    def __init__(self):
        self._client: QdrantClient | None = None
        self._encoder: SentenceTransformer | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            import atexit
            from pathlib import Path

            # Check if external Qdrant server is reachable
            use_server = False
            if settings.qdrant_url:
                try:
                    import httpx
                    r = httpx.get(f"{settings.qdrant_url}/collections", timeout=1.5)
                    if r.status_code == 200:
                        use_server = True
                except Exception:
                    use_server = False

            if use_server:
                logger.info("vector.client_server", url=settings.qdrant_url)
                self._client = QdrantClient(url=settings.qdrant_url, timeout=10)
            else:
                # Use local persistent storage directory so vectors survive CLI restarts
                storage_path = Path(__file__).resolve().parent.parent.parent / ".qdrant_storage"
                try:
                    self._client = QdrantClient(path=str(storage_path))
                    atexit.register(self._client.close)
                    logger.info("vector.client_local_storage", path=str(storage_path))
                except Exception as local_err:
                    logger.warning("vector.local_storage_failed", error=redact_secrets(str(local_err)),
                                   msg="Falling back to in-memory Qdrant")
                    self._client = QdrantClient(":memory:")
        return self._client


    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info("vector.loading_model", model=settings.embedding_model)
            self._encoder = SentenceTransformer(settings.embedding_model)
            logger.info("vector.model_loaded")
        return self._encoder

    async def ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if settings.qdrant_collection not in names:
                test_embedding = self.encoder.encode("test")
                dim = len(test_embedding)

                self.client.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    "vector.collection_created",
                    collection=settings.qdrant_collection,
                    dimension=dim,
                )
            else:
                logger.info(
                    "vector.collection_exists",
                    collection=settings.qdrant_collection,
                )
        except Exception as e:
            logger.warning("vector.ensure_collection.error", error=redact_secrets(str(e)),
                           msg="Falling back to in-memory Qdrant")
            try:
                self._client = QdrantClient(":memory:")
                test_embedding = self.encoder.encode("test")
                dim = len(test_embedding)
                self._client.create_collection(
                    collection_name=settings.qdrant_collection,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("vector.in_memory_collection_ready")
            except Exception as e2:
                logger.error("vector.fallback_failed", error=redact_secrets(str(e2)))

    def encode(self, text: str) -> list[float]:
        return self.encoder.encode(text).tolist()

    async def store_claim(
        self,
        claim_id: str,
        claim_text: str,
        verdict: str | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        embedding = self.encode(claim_text)

        payload: dict[str, Any] = {
            "claim_text": claim_text,
            "claim_id": claim_id,
            "policy_version": settings.verdict_policy_version,
        }
        if verdict:
            payload["verdict"] = verdict
        if confidence is not None:
            payload["confidence"] = confidence
        if metadata:
            payload.update(metadata)

        self.client.upsert(
            collection_name=settings.qdrant_collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, claim_id)),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        logger.debug("vector.stored", claim_id=claim_id)

    async def search_similar(
        self,
        claim_text: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        if score_threshold is None:
            score_threshold = settings.claim_similarity_threshold

        embedding = self.encode(claim_text)

        try:
            results = self.client.query_points(
                collection_name=settings.qdrant_collection,
                query=embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )

            matches = []
            for point in results.points:
                payload = point.payload or {}
                matches.append({
                    "id": str(point.id),
                    "score": point.score,
                    "claim_text": payload.get("claim_text", ""),
                    "claim_id": payload.get("claim_id", ""),
                    "verdict": payload.get("verdict"),
                    "confidence": payload.get("confidence"),
                    "policy_version": payload.get("policy_version"),
                })

            logger.debug("vector.search", query_len=len(claim_text), matches=len(matches))
            return matches

        except Exception as e:
            logger.warning("vector.search.error", error=redact_secrets(str(e)))
            return []

    async def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(settings.qdrant_collection)
            return {
                "name": settings.qdrant_collection,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", None),
                "status": info.status.value,
            }
        except Exception as e:
            return {"error": str(e)}


vector_service = VectorService()
