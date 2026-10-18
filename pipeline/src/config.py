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

    # ── Embedding Model ──────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    model_config = {"env_file": _env_file, "extra": "ignore"}


settings = Settings()

