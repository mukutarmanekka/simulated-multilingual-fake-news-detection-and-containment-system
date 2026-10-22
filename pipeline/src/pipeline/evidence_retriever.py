"""Step 4 — Real-Time Evidence Retrieval.

Generates multiple search query variants per claim and retrieves
evidence from web search, news APIs, and fact-check databases.
Assigns source credibility scores and extracts relevant passages.
"""

from __future__ import annotations

import asyncio

import structlog

from ..models.schemas import NormalizedClaim, RetrievedDocument, EvidenceRetrievalResult
from ..services.search_service import search_service
from ..services.news_service import news_service
from ..services.llm_service import llm_service
from ..utils.source_credibility import extract_domain, get_credibility, is_primary_source
from ..utils.redaction import redact_secrets
from ..config import settings

logger = structlog.get_logger()

QUERY_GEN_PROMPT = """You are a search query generator for a fact-checking system.

Given a factual claim, generate 5 diverse search queries designed to find evidence that either SUPPORTS or REFUTES the claim.

Query types to generate:
1. ORIGINAL: The claim restated as a search query
2. SIMPLIFIED: A simpler version focusing on the core fact
3. ENTITY_FOCUSED: A query centered on the main entity/person/organization
4. DATE_SPECIFIC: A query including the specific date/time if applicable
5. CONTRADICTION: A query that would find evidence contradicting the claim

Respond with JSON:
{
  "queries": [
    "query text 1",
    "query text 2",
    "query text 3",
    "query text 4",
    "query text 5"
  ]
}"""


async def retrieve_evidence(claim: NormalizedClaim) -> EvidenceRetrievalResult:
    """
    Retrieve evidence for a claim from multiple sources.

    1. Generate diverse search queries
    2. Search web (Brave/Google), news (NewsAPI)
    3. Fetch content and extract passages
    4. Score source credibility
    5. Return ≥10 candidate documents
    """
    claim_text = claim.normalized_claim or claim.original_text
    logger.info("evidence_retriever.start", claim_id=claim.claim_id)

    # ── 1. Generate search queries ────────────────────────────
    queries = await _generate_queries(claim_text, claim.resolved_entities)

    # ── 2. Search across sources in parallel ──────────────────
    all_results = []
    seen_urls = set()

    search_tasks = []
    for query in queries[:4]:  # Limit concurrent searches
        search_tasks.append(search_service.search(query, count=5))
    search_tasks.append(news_service.search_articles(claim_text, page_size=5))

    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for result_set in results:
        if isinstance(result_set, BaseException):
            logger.warning(
                "evidence_retriever.search_error",
                error=redact_secrets(str(result_set)),
            )
            continue
        for item in result_set:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(item)

    logger.info("evidence_retriever.raw_results", count=len(all_results))

    # ── 3. Build document objects with credibility ────────────
    documents = []
    content_tasks = []

    # Extract content from top URLs (limit to avoid slow processing)
    urls_to_fetch = [r.get("url") for r in all_results[:settings.max_evidence_results] if r.get("url")]
    content_tasks = [search_service.extract_content(url) for url in urls_to_fetch[:8]]
    contents = await asyncio.gather(*content_tasks, return_exceptions=True)

    content_map = {}
    for url, content in zip(urls_to_fetch[:8], contents):
        if isinstance(content, str) and content:
            content_map[url] = content

    for item in all_results[:settings.max_evidence_results]:
        url = item.get("url", "")
        domain = extract_domain(url)

        # Get source credibility
        cred = await get_credibility(url)

        # Build passages from description and extracted content
        passages = []
        desc = item.get("description") or item.get("content", "")
        if desc:
            passages.append(desc[:1000])

        full_content = content_map.get(url)
        if full_content:
            # Extract most relevant paragraphs
            relevant = _extract_relevant_passages(full_content, claim_text)
            passages.extend(relevant)

        # Determine if primary source
        entities = [e.get("resolved", e.get("original", "")) if isinstance(e, dict) else str(e)
                    for e in (claim.resolved_entities or [])]
        primary = is_primary_source(url, entities)

        doc = RetrievedDocument(
            title=item.get("title", "Untitled"),
            url=url,
            source_domain=domain,
            published_date=item.get("published_date") or item.get("publishedAt"),
            credibility_score=cred["credibility"],
            passages=passages[:3],  # Max 3 passages per doc
            retrieval_query=None,
            is_primary_source=primary,
        )
        documents.append(doc)

    logger.info(
        "evidence_retriever.done",
        claim_id=claim.claim_id,
        documents=len(documents),
        queries=len(queries),
    )

    return EvidenceRetrievalResult(
        claim_id=claim.claim_id,
        search_queries=queries,
        retrieved_documents=documents,
    )


async def _generate_queries(claim_text: str, entities: list) -> list[str]:
    """Generate diverse search queries for a claim."""
    try:
        entity_names = []
        for e in (entities or []):
            if isinstance(e, dict):
                entity_names.append(e.get("resolved", e.get("original", "")))
            else:
                entity_names.append(str(e))

        result = await llm_service.generate_json(
            system_prompt=QUERY_GEN_PROMPT,
            user_prompt=f"""Generate search queries for this claim:
"{claim_text}"

Key entities: {', '.join(entity_names) if entity_names else 'None identified'}""",
            temperature=0.3,
            max_tokens=1024,
        )
        
        raw_queries = result.get("queries", [claim_text])
        parsed_queries = []
        
        # Robustly parse queries in case the LLM returns dicts instead of strings
        for q in raw_queries:
            if isinstance(q, dict):
                # Extract from common keys LLMs might use
                q_str = q.get("query") or q.get("text") or q.get("value") or ""
                if q_str:
                    parsed_queries.append(str(q_str))
            elif isinstance(q, str) and q.strip():
                parsed_queries.append(q.strip())
                
        if not parsed_queries:
            parsed_queries = [claim_text]
            
        return parsed_queries[:5]

    except Exception as e:
        logger.warning(
            "evidence_retriever.query_gen_error",
            error=redact_secrets(str(e)),
        )
        # Fallback: use the claim text directly
        return [claim_text]


def _extract_relevant_passages(content: str, claim_text: str, max_passages: int = 2) -> list[str]:
    """Extract the most relevant passages from document content."""
    if not content:
        return []

    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 50]

    if not paragraphs:
        return [content[:500]] if content else []

    # Simple keyword overlap scoring
    claim_words = set(claim_text.lower().split())
    scored = []
    for para in paragraphs:
        para_words = set(para.lower().split())
        overlap = len(claim_words & para_words)
        scored.append((overlap, para))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [para[:800] for _, para in scored[:max_passages]]
