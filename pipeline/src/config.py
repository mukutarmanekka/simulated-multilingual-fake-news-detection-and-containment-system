from pathlib import Path
from pydantic_settings import BaseSettings

# Dynamically locate .env
_curr = Path(__file__).resolve().parent
_candidates = [
    _curr.parent.parent / ".env",  # repo root / .env
    _curr.parent / ".env",         # pipeline / .env
    Path(".env"),                  # cwd / .env
    Path("../.env"),               # parent / .env
]
_env_file = next((str(p) for p in _candidates if p.is_file()), ".env")


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # ── Database ──────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://fnews:fnews_secret@localhost:5432/fakenews"

    # ── Redis ─────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "claims"

    # ── LLM ───────────────────────────────────────────────────
    llm_provider: str = "gemini"  # "gemini", "deepseek", "openai", "openai-compatible", "ollama", or "mock"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    openai_model: str = "gpt-4o"
    deepseek_model: str = "deepseek-chat"
    openai_compatible_base_url: str = "http://localhost:1234/v1"
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # ── Search ────────────────────────────────────────────────
    brave_search_api_key: str = ""
    tavily_api_key: str = ""
    google_cse_api_key: str = ""
    google_cse_cx: str = ""

    # ── Fact-Check ────────────────────────────────────────────
    google_factcheck_api_key: str = ""

    # ── News ──────────────────────────────────────────────────
    newsapi_key: str = ""

    # ── Pipeline Tuning ───────────────────────────────────────
    claim_similarity_threshold: float = 0.88
    confidence_escalation_threshold: float = 0.65
    max_evidence_results: int = 10
    top_evidence_count: int = 5

    # ── Tier 4 Source Blacklist ───────────────────────────────
    # Empty means pipeline/config/blacklisted_sources.json.
    blacklist_config_path: str = ""

    # ── Internal Vector Cache ─────────────────────────────────
    # Stamped into cached claims so old-policy verdicts are not reused.
    verdict_policy_version: str = "verdict_policy_v2"

    # ── User-Facing Verdict Policy ────────────────────────────
    # Cached/fast-path verdicts use these confidence thresholds.
    verdict_real_min_confidence: float = 0.65
    verdict_fake_min_confidence: float = 0.65

    # Evidence-based verdicts use a calibrated support score built from
    # credible/relevant documents, LLM subclaim alignment, and NLI stance.
    # The confidence floor prevents a nearly-random verifier from passing.
    verdict_confidence_floor: float = 0.45
    verdict_evidence_strength_threshold: float = 0.75
    verdict_subclaim_alignment_threshold: float = 0.60
    verdict_stance_alignment_threshold: float = 0.60
    verdict_stance_conflict_count: int = 2

    verdict_min_source_credibility: float = 0.60
    verdict_strong_source_credibility: float = 0.80
    verdict_min_relevance: float = 0.30
    verdict_min_corroborating_sources: int = 2

    # ── Embedding Model ──────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    model_config = {"env_file": _env_file, "extra": "ignore"}


settings = Settings()

