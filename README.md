# VeriFacto — Real-Time Fake News Detection System

A production-grade, **multilingual**, evidence-based claim verification pipeline that extracts factual claims from news content in **50+ languages**, auto-translates to English, retrieves real-time evidence from trusted sources, and produces auditable verdicts with calibrated confidence scores.

---

## Architecture

```
                    Terminal CLI (cli.py)
                           │
                           ▼
          ┌───────────────────────────────────┐
          │   Language Detection & Translate  │
          │   (langdetect + Gemini Fallback)  │
          └─────────────────┬─────────────────┘
                            │
                            ▼
          ┌───────────────────────────────────┐
          │        9-Step ML Pipeline         │
          │  ┌─────────────────────────────┐  │
          │  │ 1. Claim Extraction (LLM)   │  │
          │  │ 2. Claim Normalization      │  │
          │  │ 3. Fast-Path Match (Vector) │  │
          │  │ 4. Evidence Retrieval       │◄─┼─── Tavily Search API
          │  │ 5. Cross-Encoder Reranker   │  │    Google Fact Check API
          │  │ 6. NLI Stance + Verifier    │◄─┼─── Google Gemini API
          │  │ 7. Calibrated Confidence    │  │    (gemini-2.5-flash-lite)
          │  │ 8. Human Escalation Check   │  │
          │  │ 9. Vector & Audit Storage   │  │
          │  └──────────────┬──────────────┘  │
          └─────────────────┼─────────────────┘
                            ▼
               ┌─────────────────────────┐
               │   Persistent Storage    │
               │  • PostgreSQL (Claims)  │
               │  • Qdrant (Disk/Memory) │
               └─────────────────────────┘
```

---

## Key Features

* **Multilingual First**: Detects over 50 languages (`hi`, `fr`, `es`, `de`, `ar`, `zh`, etc.) and seamlessly auto-translates to English with multi-tier fallback (Google Translate + Gemini API).
* **Parallel Multi-Claim Processing**: Analyzes articles containing multiple claims concurrently via asynchronous execution.
* **Fast-Path Caching**: Matches newly submitted claims against previously verified claims using local disk-backed Qdrant vector embeddings and Google's ClaimReview database.
* **Neural Evidence Reranking**: Re-ranks search results using `cross-encoder/ms-marco-MiniLM-L-6-v2` based on semantic relevance, source credibility, and temporal freshness.
* **NLI Stance Classification**: DeBERTa cross-encoder (`cross-encoder/nli-deberta-v3-base`) assigns `SUPPORTS`, `REFUTES`, or `NEUTRAL` stance to every retrieved passage.
* **Standalone or Containerized**: Runs 100% locally with automatic in-memory/disk vector fallbacks, or connects to external PostgreSQL and Qdrant containers via Docker Compose.

---

## Quick Start

### Prerequisites

* **Python 3.11+** (Python 3.13 supported)
* **API Keys** (Set in `.env`):
  * `GEMINI_API_KEY`: For LLM claim extraction, normalization, and verification.
  * `TAVILY_API_KEY`: For real-time web evidence retrieval.
  * `GOOGLE_FACTCHECK_API_KEY`: (Recommended) For fast-path matching against existing fact-checks.
* **Optional**: Docker / Docker Compose (if you prefer external PostgreSQL & Qdrant servers).

---

### Step 1: Configure Environment

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here

TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_FACTCHECK_API_KEY=your_google_factcheck_key_here
```

> **DeepSeek API:** To use DeepSeek's hosted API, set `LLM_PROVIDER=deepseek`, provide your `DEEPSEEK_API_KEY`, and optionally specify `DEEPSEEK_MODEL=deepseek-chat` (or `deepseek-flash`). DeepSeek automatically connects to `https://api.deepseek.com` and requests structured JSON output.

> **Local LLM Alternative (Ollama):** If you prefer running local models, set `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3:8b` (or `deepseek-r1:8b`), and ensure Ollama is running on `http://localhost:11434`. If Ollama is unreachable, the system automatically falls back to Gemini or mock mode.

> **Custom OpenAI-compatible API:** To use a hosted or local model exposing the OpenAI Chat Completions API, set `LLM_PROVIDER=openai-compatible`, `OPENAI_COMPATIBLE_BASE_URL` (including `/v1`), and `OPENAI_COMPATIBLE_MODEL`. Set `OPENAI_COMPATIBLE_API_KEY` if the server requires authentication; it can be blank for local servers. Requests use the `/chat/completions` endpoint and request JSON-mode output.

---

### Step 2: Install Python Dependencies

```bash
cd pipeline
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

*(Note: On the first run, ~1.5GB of HuggingFace models for sentence embeddings and NLI are automatically downloaded and cached).*

---

### Step 3: (Optional) Start External Infrastructure

The pipeline automatically uses persistent disk vector storage (`.qdrant_storage`) if external Qdrant is not running. However, if you want dedicated PostgreSQL and Qdrant services:

```bash
# From the project root:
docker compose up -d
```

---

### Step 4: Run the CLI

#### 1. Interactive Terminal Mode
```bash
python cli.py
```
Type any claim or headline in English, Hindi, Spanish, etc. Type `/history` to review past claims, or `/quit` to exit.

#### 2. Single-Shot Mode (English)
```bash
python cli.py "The Eiffel Tower is located in Paris, France."
```

#### 3. Single-Shot Mode (Multilingual)
```bash
# Hindi claim (Chandrayaan-3 Moon landing)
python cli.py "भारत ने चंद्रयान-3 मिशन को चंद्रमा के दक्षिणी ध्रुव पर सफलतापूर्वक उतारा।"

# French claim
python cli.py "La NASA a confirmé la présence de glace d'eau sur la Lune."
```

#### 4. View Verification History
```bash
python cli.py --history
```

---

## Real CLI Output Example

```text
+==============================================================+
|         Real-Time Fake News Detection Pipeline               |
|                    Terminal Interface                         |
+==============================================================+

  Initializing...  [OK] Ready.          

  [...] Verifying: "The Eiffel Tower is located in Paris, France."...

--------------------------------------------------------------
  Claim:  The Eiffel Tower is located in Paris, France.

    [SUPPORTED]     Confidence: ################.... 80%

  Reasoning:
    > Step 1: The claim asserts that the Eiffel Tower is located in Paris, France.
    > Step 2: Evidence from toureiffel.paris states 'The Eiffel Tower is located in the heart of Paris' and lists its address as '5 avenue Anatole France, 75007 Paris, France.'
    > Step 3: Evidence from Wikipedia corroborates that 'The Eiffel Tower... is a lattice tower on the Champ de Mars in Paris, France.'
    > Step 4: All parts of the claim are directly confirmed by multiple pieces of evidence without contradiction, leading to a verdict of SUPPORTED.

  Evidence (5 sources):
    [+] toureiffel.paris
      The Eiffel Tower is located in the heart of Paris, in the 7th arrondissement, on the Champ de Mars...
      https://www.toureiffel.paris/en/access-map
    [+] britannica.com
      The Eiffel Tower can be found on the Champs de Mars at 5 Avenue Anatole...
      https://www.britannica.com/topic/Eiffel-Tower-Paris-France
    [+] en.wikipedia.org
      The Eiffel Tower is a lattice tower on the Champ de Mars in Paris, France...
      https://en.wikipedia.org/wiki/Eiffel_Tower

  38.4s
--------------------------------------------------------------

  Total: 1 claim(s) verified in 42.7s
```

---

## Running the Automated Test Suite

The project includes unit and integration tests covering language detection, confidence scoring, human escalation, and API endpoints:

```bash
cd pipeline
pytest tests/
```

Expected output:
```text
collected 13 items
tests\test_api.py ..                                                     [ 15%]
tests\test_confidence.py ...                                             [ 38%]
tests\test_escalation.py ...                                             [ 61%]
tests\test_language.py .....                                             [100%]
============================= 13 passed in 10.37s =============================
```

---

## 9-Step Verification Pipeline

1. **Claim Extraction**: Extracts atomic checkable assertions using Gemini Flash + spaCy NER.
2. **Claim Normalization**: Resolves pronouns, relative dates (e.g. "yesterday" → ISO dates), quantities, and canonical entity names.
3. **Claim Matching**: Checks local vector embeddings and Google ClaimReview for existing verdicts (fast path).
4. **Evidence Retrieval**: Generates 5 diverse search queries and retrieves documents via Tavily API and live web extraction.
5. **Evidence Re-ranking**: Scores retrieved passages with `cross-encoder/ms-marco-MiniLM-L-6-v2` based on relevance, credibility, freshness, and primary source weighting.
6. **Claim Verification**: Performs atomic sub-claim verification and chain-of-thought evidence synthesis with `cross-encoder/nli-deberta-v3-base`.
7. **Confidence Estimation**: Computes a calibrated confidence score (0.00 – 1.00) based on source agreement, contradiction penalties, and evidence quality.
8. **Human Escalation**: Flags high-impact domains (health, war, elections, public disasters) with low confidence for human review.
9. **Persistence**: Saves claims, evidence, stances, and audit traces to PostgreSQL and Qdrant.

---

## Tech Stack

| Component         | Technology / Model                                   |
|-------------------|------------------------------------------------------|
| **CLI Engine**    | Python 3.11+ / 3.13, `asyncio`, `argparse`          |
| **LLM Provider**  | Google Gemini (`gemini-2.5-flash-lite`) / Ollama (`llama3`) |
| **Embeddings**    | `sentence-transformers/all-MiniLM-L6-v2` (384-dim)   |
| **Reranker**      | `cross-encoder/ms-marco-MiniLM-L-6-v2`               |
| **NLI Stance**    | `cross-encoder/nli-deberta-v3-base`                  |
| **Vector DB**     | Qdrant (Persistent local disk or container)          |
| **Relational DB** | PostgreSQL 16 (Async SQLAlchemy + asyncpg)           |
| **Search API**    | Tavily Search API                                    |
| **Fact-Check DB** | Google Fact Check Tools API                          |
| **Language**      | `langdetect` + `deep-translator` + Gemini Fallback   |
| **Test Suite**    | `pytest`, `pytest-asyncio`                           |

---

## Project Structure

```
RealTimeFakeNewsDetection/
├── pipeline/
│   ├── cli.py                    # Terminal CLI entry point
│   ├── requirements.txt          # Python dependencies
│   ├── .qdrant_storage/          # Persistent local vector database
│   ├── tests/                    # Automated test suite
│   │   ├── test_api.py
│   │   ├── test_confidence.py
│   │   ├── test_escalation.py
│   │   └── test_language.py
│   └── src/
│       ├── main.py               # FastAPI application
│       ├── config.py             # Settings & dynamic .env resolution
│       ├── api/routes.py         # HTTP endpoints
│       ├── pipeline/
│       │   ├── orchestrator.py   # Pipeline controller (concurrent execution)
│       │   ├── claim_extractor.py
│       │   ├── claim_normalizer.py
│       │   ├── claim_matcher.py
│       │   ├── evidence_retriever.py
│       │   ├── evidence_reranker.py
│       │   ├── claim_verifier.py
│       │   ├── confidence.py
│       │   └── escalation.py
│       ├── services/
│       │   ├── llm_service.py    # LLM provider with auto-fallback
│       │   ├── vector_service.py # Qdrant disk & server client
│       │   ├── search_service.py # Tavily / web scraping
│       │   ├── factcheck_service.py
│       │   └── news_service.py
│       ├── utils/
│       │   ├── language.py       # Language detection & translation
│       │   ├── source_credibility.py
│       │   └── temporal.py
│       ├── db/postgres.py        # Database operations
│       └── models/schemas.py     # Pydantic schemas
├── db/
│   └── init.sql                  # PostgreSQL schema definition
├── docker-compose.yml            # Docker Compose configuration
├── .env                          # Active configuration
├── .env.example                  # Template configuration
└── README.md
```

---

## License

MIT
