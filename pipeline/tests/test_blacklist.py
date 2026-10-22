import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.enums import InputType, Stance, UserVerdict, Verdict
from src.models.schemas import (
    ClaimExtractionResult,
    ClaimMatchResult,
    ConfidenceBreakdown,
    EscalationDecision,
    EvidenceRetrievalResult,
    EvidenceStance,
    ExtractedClaim,
    FinalClaimResponse,
    NormalizedClaim,
    RankedEvidence,
    RerankedEvidenceResult,
    VerificationJobResponse,
    VerificationResult,
    VerifyRequest,
)
from src.pipeline import orchestrator as orchestrator_module
from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.blacklist import (
    match_blacklisted_source,
    normalize_hostname,
    reload_blacklist,
)
from src.utils.source_credibility import (
    extract_domain,
    get_credibility,
    is_primary_source,
)
from src.utils.language import LanguageResult


class FakeDB:
    """In-memory stand-in for ClaimDB so tests never touch PostgreSQL."""

    def __init__(self):
        self.calls = []

    async def create_job(self, *args, **kwargs):
        self.calls.append("create_job")

    async def update_job_status(self, *args, **kwargs):
        self.calls.append("update_job_status")

    async def store_claim(self, *args, **kwargs):
        self.calls.append("store_claim")

    async def log_audit(self, *args, **kwargs):
        self.calls.append("log_audit")

    async def store_verification(self, *args, **kwargs):
        return "verification-1"

    async def store_evidence(self, *args, **kwargs):
        return "evidence-1"

    async def create_escalation(self, *args, **kwargs):
        return "review-1"


def make_orchestrator() -> PipelineOrchestrator:
    orchestrator = PipelineOrchestrator()
    orchestrator._db = FakeDB()
    return orchestrator


# ── 1-7: Domain normalization and matching ───────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "https://theonion.com/article",
        "http://www.theonion.com/article",
        "theonion.com",
        "www.theonion.com/news",
        "https://theonion.com:8443/path?q=1#fragment",
        "https://user:pass@theonion.com/path",
        "THEONION.COM",
        "https://theonion.com./path",
    ],
)
def test_blacklisted_domain_matches(value):
    match = match_blacklisted_source(value)
    assert match is not None
    assert match.matched_domain == "theonion.com"
    assert match.category == "satire"
    assert match.reason == "BLACKLISTED_SATIRE_SOURCE"
    assert match.tier == "BLACKLISTED"
    assert match.tier_number == 4


def test_www_and_non_www_match_same_domain():
    with_www = match_blacklisted_source("https://www.theonion.com/a")
    without_www = match_blacklisted_source("https://theonion.com/a")
    assert with_www is not None
    assert without_www is not None
    assert with_www.matched_domain == without_www.matched_domain


def test_legitimate_subdomain_matches_by_label_suffix():
    match = match_blacklisted_source("https://satire.theonion.com/article")
    assert match is not None
    assert match.matched_domain == "theonion.com"
    assert match.input_host == "satire.theonion.com"

    fauxy_subdomain = match_blacklisted_source("https://gujarati.thefauxy.com/news")
    assert fauxy_subdomain is not None
    assert fauxy_subdomain.matched_domain == "thefauxy.com"


def test_fauxy_canonical_and_alternate_domains_match():
    for value in (
        "https://thefauxy.com/article",
        "https://fauxy.com/article",
        "https://gujarati.thefauxy.com/news",
    ):
        match = match_blacklisted_source(value)
        assert match is not None
        assert match.category == "satire"
        assert match.reason == "BLACKLISTED_SATIRE_SOURCE"


@pytest.mark.parametrize(
    "value",
    [
        "https://theonion.com.evil.example/article",
        "https://nottheonion.com/article",
        "https://theonion.company/article",
        "https://theonion.com.br/article",
        "https://theonion.com@evil.example/article",
        "https://evil.example/?next=https://theonion.com",
        "https://evil.example/theonion.com/article",
        "https://evil.example#theonion.com",
    ],
)
def test_deceptive_domains_do_not_match(value):
    assert match_blacklisted_source(value) is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "http://",
        "https://[::1",
        "not a url at all",
        "javascript:alert(1)",
        "%%%",
    ],
)
def test_malformed_or_missing_urls_do_not_crash(value):
    assert normalize_hostname(value) is None or isinstance(normalize_hostname(value), str)
    assert match_blacklisted_source(value) is None


def test_url_components_are_stripped():
    assert normalize_hostname("https://www.theonion.com:443/a/b?x=1#frag") == "theonion.com"
    assert extract_domain("https://www.theonion.com:8443/a?b=1#c") == "theonion.com"
    assert extract_domain("not a url") == ""


def test_registry_is_configurable(tmp_path):
    config = tmp_path / "blacklist.json"
    config.write_text(
        json.dumps(
            {
                "domains": {
                    "example.test": {
                        "name": "Example",
                        "category": "troll",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        reload_blacklist(config)
        match = match_blacklisted_source("https://news.example.test/story")
        assert match is not None
        assert match.matched_domain == "example.test"
        assert match.category == "troll"
        assert match.reason == "BLACKLISTED_TROLL_SOURCE"
    finally:
        reload_blacklist()


def test_missing_registry_file_is_non_fatal(tmp_path):
    try:
        reload_blacklist(tmp_path / "does-not-exist.json")
        assert match_blacklisted_source("https://theonion.com/a") is None
    finally:
        reload_blacklist()


def test_single_label_registry_entry_is_rejected(tmp_path):
    config = tmp_path / "blacklist.json"
    config.write_text(
        json.dumps(
            {
                "domains": {
                    "com": "fake_news",
                    "good.test": "satire",
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        reload_blacklist(config)
        # "com" must not suffix-match every .com host.
        assert match_blacklisted_source("https://anything.com/story") is None
        assert match_blacklisted_source("https://good.test/story") is not None
    finally:
        reload_blacklist()


@pytest.mark.asyncio
async def test_source_credibility_marks_tier_4():
    credibility = await get_credibility("https://www.theonion.com/news")
    assert credibility["tier"] == "BLACKLISTED"
    assert credibility["tier_number"] == 4
    assert credibility["credibility"] == 0.0
    assert credibility["blacklisted"] is True
    assert credibility["category"] == "satire"
    assert credibility["reason"] == "BLACKLISTED_SATIRE_SOURCE"


def test_blacklisted_source_is_never_primary():
    assert is_primary_source("https://theonion.com/press-release", []) is False


# ── 1-2, 8: Orchestrator behavior ────────────────────────────────────


@pytest.mark.asyncio
async def test_blacklisted_source_returns_fake_without_detector_calls(monkeypatch):
    extract = AsyncMock()
    retrieve = AsyncMock()
    verify = AsyncMock()
    translate = MagicMock()

    llm_generate = AsyncMock()
    search = AsyncMock()
    news = AsyncMock()
    factcheck = AsyncMock()

    monkeypatch.setattr(orchestrator_module, "extract_claims", extract)
    monkeypatch.setattr(orchestrator_module, "retrieve_evidence", retrieve)
    monkeypatch.setattr(orchestrator_module, "verify_claim", verify)
    monkeypatch.setattr(orchestrator_module, "detect_and_translate", translate)
    monkeypatch.setattr(
        "src.services.llm_service.llm_service.generate_json", llm_generate
    )
    monkeypatch.setattr("src.services.search_service.search_service.search", search)
    monkeypatch.setattr(
        "src.services.news_service.news_service.search_articles", news
    )
    monkeypatch.setattr(
        "src.services.factcheck_service.factcheck_service.search_claims", factcheck
    )

    orchestrator = PipelineOrchestrator()
    fake_db = FakeDB()
    orchestrator._db = fake_db

    result = await orchestrator.run(
        input_text="A fabricated story published by a satirical outlet.",
        source_url="https://www.theonion.com/some-article",
    )

    assert isinstance(result, VerificationJobResponse)
    assert result.total_claims == 1
    assert len(result.results) == 1

    claim_result = result.results[0]
    assert isinstance(claim_result, FinalClaimResponse)
    assert claim_result.verdict == UserVerdict.FAKE
    assert claim_result.verdict.value == "FAKE"
    assert claim_result.verdict_reason == "BLACKLISTED_SATIRE_SOURCE"
    assert claim_result.blacklist_category == "satire"
    assert claim_result.matched_blacklist_domain == "theonion.com"
    assert claim_result.source_domain == "theonion.com"
    assert claim_result.source_tier == "BLACKLISTED"
    assert claim_result.source_tier_number == 4
    assert claim_result.evidence_verdict is None
    assert result.source_url == "https://www.theonion.com/some-article"

    extract.assert_not_awaited()
    retrieve.assert_not_awaited()
    verify.assert_not_awaited()
    assert translate.call_count == 0
    llm_generate.assert_not_awaited()
    search.assert_not_awaited()
    news.assert_not_awaited()
    factcheck.assert_not_awaited()

    # The short-circuit also happens before any database work.
    assert fake_db.calls == []


@pytest.mark.asyncio
async def test_input_type_url_treats_bare_url_as_source(monkeypatch):
    extract = AsyncMock()
    translate = MagicMock()

    monkeypatch.setattr(orchestrator_module, "extract_claims", extract)
    monkeypatch.setattr(orchestrator_module, "detect_and_translate", translate)

    orchestrator = make_orchestrator()
    result = await orchestrator.run(
        input_text="https://www.theonion.com/some-article",
        input_type=InputType.URL,
    )

    assert result.results[0].verdict == UserVerdict.FAKE
    assert result.results[0].source_domain == "theonion.com"
    assert result.results[0].matched_blacklist_domain == "theonion.com"
    extract.assert_not_awaited()
    assert translate.call_count == 0


@pytest.mark.asyncio
async def test_blacklisted_domain_mentioned_in_article_does_not_short_circuit(monkeypatch):
    extraction = ClaimExtractionResult(
        claims=[],
        non_checkable_statements=["A news article that merely mentions a website."],
    )
    extract = AsyncMock(return_value=extraction)
    language_result = LanguageResult(
        original_text="An article that mentions a blacklisted domain.",
        processed_text="An article that mentions a blacklisted domain.",
        detected_language="en",
        language_name="English",
        was_translated=False,
        translation_failed=False,
    )

    monkeypatch.setattr(orchestrator_module, "extract_claims", extract)
    monkeypatch.setattr(
        orchestrator_module,
        "detect_and_translate",
        MagicMock(return_value=language_result),
    )

    orchestrator = make_orchestrator()
    result = await orchestrator.run(
        input_text=(
            "A news article mentions https://theonion.com/some-article "
            "but is not published by that outlet."
        )
    )

    extract.assert_awaited_once()
    assert result.total_claims == 0
    assert result.results == []
    assert result.source_url is None


@pytest.mark.asyncio
async def test_non_blacklisted_source_continues_through_pipeline(monkeypatch):
    claim = ExtractedClaim(
        claim_id="claim-1",
        text="The Eiffel Tower is located in Paris.",
        entities=["Eiffel Tower", "Paris"],
    )
    extraction = ClaimExtractionResult(claims=[claim], non_checkable_statements=[])
    normalized = NormalizedClaim(
        claim_id=claim.claim_id,
        original_text=claim.text,
        normalized_claim=claim.text,
    )
    match_result = ClaimMatchResult(matched_existing_claim=False)
    raw_evidence = EvidenceRetrievalResult(
        claim_id=claim.claim_id,
        retrieved_documents=[],
    )
    ranked_document = RankedEvidence(
        doc_id="doc-1",
        title="Eiffel Tower location",
        url="https://example.com/eiffel-tower",
        source_domain="example.com",
        published_date="2026-01-01",
        credibility_score=0.90,
        relevance_score=0.90,
        freshness_score=0.80,
        final_rank_score=0.90,
        passages=["The Eiffel Tower is located in Paris, France."],
        is_primary_source=False,
    )
    ranked_evidence = RerankedEvidenceResult(
        claim_id=claim.claim_id,
        ranked_evidence=[ranked_document],
    )
    verification = VerificationResult(
        claim_id=claim.claim_id,
        verdict=Verdict.SUPPORTED,
        confidence=0.0,
        reasoning=["The evidence confirms the claim."],
        supporting_evidence=[
            EvidenceStance(
                doc_id="doc-1",
                snippet="The Eiffel Tower is located in Paris.",
                stance=Stance.SUPPORTS,
                confidence=0.95,
            )
        ],
    )
    escalation = EscalationDecision(requires_human_review=False)
    language_result = LanguageResult(
        original_text=claim.text,
        processed_text=claim.text,
        detected_language="en",
        language_name="English",
        was_translated=False,
        translation_failed=False,
    )

    extract = AsyncMock(return_value=extraction)
    normalize = AsyncMock(return_value=normalized)
    match = AsyncMock(return_value=match_result)
    retrieve = AsyncMock(return_value=raw_evidence)
    rerank = AsyncMock(return_value=ranked_evidence)
    verify = AsyncMock(return_value=verification)
    confidence = AsyncMock(
        return_value=(0.85, ConfidenceBreakdown(final_confidence=0.85))
    )
    escal = AsyncMock(return_value=escalation)

    monkeypatch.setattr(orchestrator_module, "extract_claims", extract)
    monkeypatch.setattr(orchestrator_module, "normalize_claim", normalize)
    monkeypatch.setattr(orchestrator_module, "match_claim", match)
    monkeypatch.setattr(orchestrator_module, "retrieve_evidence", retrieve)
    monkeypatch.setattr(orchestrator_module, "rerank_evidence", rerank)
    monkeypatch.setattr(orchestrator_module, "verify_claim", verify)
    monkeypatch.setattr(orchestrator_module, "estimate_confidence", confidence)
    monkeypatch.setattr(orchestrator_module, "check_escalation", escal)
    monkeypatch.setattr(
        orchestrator_module,
        "detect_and_translate",
        MagicMock(return_value=language_result),
    )

    orchestrator = make_orchestrator()
    monkeypatch.setattr(orchestrator, "_store_results", AsyncMock())

    result = await orchestrator.run(
        input_text="The Eiffel Tower is located in Paris.",
        source_url="https://example.com/eiffel-tower",
    )

    extract.assert_awaited_once()
    normalize.assert_awaited_once()
    match.assert_awaited_once()
    retrieve.assert_awaited_once()
    rerank.assert_awaited_once()
    verify.assert_awaited_once()
    confidence.assert_awaited_once()
    escal.assert_awaited_once()

    assert result.results[0].verdict == UserVerdict.REAL
    assert result.results[0].verdict_reason == "EVIDENCE_CORROBORATES_CLAIM"
    assert result.results[0].evidence_verdict == Verdict.SUPPORTED
    assert result.results[0].source_domain == "example.com"


# ── API schema/plumbing ──────────────────────────────────────────────


def test_verify_request_accepts_source_url_and_url_alias():
    request = VerifyRequest(
        text="A test article body.",
        source_url="https://theonion.com/article",
    )
    assert request.source_url == "https://theonion.com/article"

    alias_request = VerifyRequest(
        text="A test article body.",
        url="https://theonion.com/article",
    )
    assert alias_request.source_url == "https://theonion.com/article"


@pytest.mark.asyncio
async def test_api_route_passes_source_url(monkeypatch):
    from src.api import routes

    captured = {}

    class FakeOrchestrator:
        async def run(self, **kwargs):
            captured.update(kwargs)
            return VerificationJobResponse(
                job_id="job-1",
                input_text=kwargs["input_text"],
                input_type=kwargs["input_type"],
            )

    monkeypatch.setattr(routes, "get_orchestrator", lambda: FakeOrchestrator())

    request = VerifyRequest(
        text="A test article body.",
        input_type=InputType.ARTICLE,
        source_url="https://theonion.com/article",
    )
    await routes.verify_content(request)

    assert captured["source_url"] == "https://theonion.com/article"
