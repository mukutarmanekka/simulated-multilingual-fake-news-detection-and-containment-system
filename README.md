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

## Source Policy: Tier 4 Blacklist

The detector has a dedicated Tier 4 for configured publisher-level policy
matches. It is stored separately from the ordinary credibility tiers and is
loaded from [`pipeline/config/blacklisted_sources.json`](pipeline/config/blacklisted_sources.json)
by `pipeline/src/utils/blacklist.py`.

Tier 4 is **not** a universal truth oracle. It records that a publisher is
configured as satire, parody, troll, or a known fake-news outlet. Satire and
fabricated news are different categories, and the response exposes the
category and reason code so callers can present that distinction.

When a request includes `source_url` (or a bare domain), the orchestrator
normalizes the hostname and checks Tier 4 **before** language translation,
claim extraction, database work, search, or any LLM/API call. A match returns:

- verdict `FAKE`
- `verdict_reason` such as `BLACKLISTED_SATIRE_SOURCE`
- `matched_blacklist_domain`, `blacklist_category`, `source_tier`
  (`BLACKLISTED`), and `source_tier_number` (`4`)
- `evidence_verdict = null` because no evidence pipeline ran

Matching rules:

- Exact domains and complete-label subdomains match: `theonion.com` and
  `satire.theonion.com` match; `theonion.com.evil.example` and
  `nottheonion.com` do not.
- `www.`, schemes, ports, paths, query strings, fragments, userinfo, trailing
  dots, and IDN hostnames are normalized safely.
- A malformed or missing URL never crashes the detector; it simply bypasses
  the blacklist.
- The blacklist is only checked against an explicitly supplied source URL. It
  is never inferred from a website merely mentioned inside the article text.
  One exception is explicit: when `input_type` is `url` and the entire `text`
  is a single URL, that URL is treated as the source.

### Adding or removing domains

Edit `pipeline/config/blacklisted_sources.json`. The dictionary key is the
canonical domain; the value may be an object or a category string:

```json
{
  "domains": {
    "example-satire.com": {
      "name": "Example Satire",
      "category": "satire",
      "notes": "Optional note"
    },
    "example-fake.com": "fake_news"
  }
}
```

Allowed categories: `satire`, `parody`, `troll`, `fake_news`. Restart the
process (or call `reload_blacklist()`) after editing. Set
`BLACKLIST_CONFIG_PATH` to use a different registry file.

---

## User-Facing Verdicts

Every claim result returns exactly one user-facing verdict:

| Verdict | Meaning | Decision rule |
|---|---|---|
| `REAL` | The claim is corroborated. | Internal `SUPPORTED` + confidence ≥ 0.45 + evidence support score ≥ 0.75 + credible/relevant documents + either LLM subclaim alignment ≥ 0.60 or NLI stance alignment ≥ 0.60; no conflicting subclaims or opposite-stance majority. Or an authoritative fact-check match confirms it. |
| `FAKE` | The source is Tier 4, or reliable evidence contradicts the claim. | Tier 4 match (immediate), or internal `REFUTED` with the same evidence gates and refuting subclaims/stances; or an authoritative fact-check match refutes it. |
| `UNCERTAIN` | Evidence is insufficient, conflicting, irrelevant, unavailable, or weak. | `NOT_ENOUGH_EVIDENCE`, `CONFLICTING`, low confidence, subclaim/stance conflicts, irrelevant or low-credibility evidence, or an insufficient support score. |

The evidence support score is:

- **55%** document quality: at least two credible, relevant documents, or
  one primary / credibility ≥ 0.80 document.
- **35%** LLM subclaim alignment: the share of verifier subclaims matching
  the overall `SUPPORTED`/`REFUTED` decision. The verifier prompt requires
  the first subclaim to be the central assertion and forbids background-fact
  subclaims. For `REFUTED`, a refuted central subclaim is sufficient;
  supported auxiliary subclaims do not block `FAKE`. For `SUPPORTED`, any
  refuted subclaim blocks `REAL`.
- **10%** NLI stance alignment. NLI is deliberately secondary because it
  often returns `NEUTRAL` for paraphrased or indirect evidence; it is not
  treated as contradiction and is not the sole gate.

Source credibility is a **publisher prior**, not proof that a particular
article supports a claim. Article-level relevance, subclaim verdicts, and
NLI stance decide support. A single arbitrary source is not enough; one
primary or very-high-credibility source can qualify.

The built-in registry includes official primary publishers (such as the
official Eiffel Tower site), edited reference works (Britannica), a moderate
prior for crowd-edited Wikipedia (0.65), established history/educational
publishers, and low priors for user-generated and social sources. Unknown
domains default to 0.50. Tier 4 remains a separate, explicitly configured
policy rule and always wins.

Thresholds are configurable in `.env`:
`VERDICT_CONFIDENCE_FLOOR`, `VERDICT_EVIDENCE_STRENGTH_THRESHOLD`,
`VERDICT_SUBCLAIM_ALIGNMENT_THRESHOLD`, `VERDICT_STANCE_ALIGNMENT_THRESHOLD`,
`VERDICT_STANCE_CONFLICT_COUNT`, `VERDICT_MIN_SOURCE_CREDIBILITY`,
`VERDICT_STRONG_SOURCE_CREDIBILITY`, `VERDICT_MIN_RELEVANCE`, and
`VERDICT_MIN_CORROBORATING_SOURCES`. The fast-path cached-verdict thresholds
remain `VERDICT_REAL_MIN_CONFIDENCE` / `VERDICT_FAKE_MIN_CONFIDENCE`.

The policy is evaluated on `pipeline/data/validation_claims.json` with
`pipeline/scripts/validate_verdict_policy.py`. Use `--sweep` to reproduce
the threshold calibration. This is a policy-level labelled set, not an
end-to-end accuracy claim; the opt-in live suite provides the end-to-end
check.

The internal four-way verdict (`SUPPORTED`/`REFUTED`/`NOT_ENOUGH_EVIDENCE`/
`CONFLICTING`) is preserved in `evidence_verdict` and in the database. Only
the user-facing `verdict` field is reduced to the three values above.

### Passing a source URL to the API

```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article text to verify",
    "source_url": "https://www.theonion.com/some-article"
  }'
```

`url` is also accepted as an alias for `source_url`. Omitting it keeps the
existing behavior unchanged.

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

#### 4. Single-Shot Mode with a Publisher URL
```bash
# The URL is checked against the Tier 4 blacklist before any detector,
# search, or LLM call. A match returns FAKE immediately.
python cli.py --source-url "https://www.theonion.com/some-article" "Article text to verify"

# A bare domain works too
python cli.py --source-url "thefauxy.com" "Article text to verify"
```

In interactive mode, use `/source URL` to set the publisher URL and
`/source clear` to remove it.

#### 5. View Verification History
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

    [REAL]          Confidence: ################.... 80%

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

The project includes unit, integration, and failure-path tests covering
language detection, confidence scoring, human escalation, API endpoints,
Tier 4 matching, the blacklist short-circuit, the REAL/FAKE/UNCERTAIN
verdict policy, NLI label mapping, provider timeouts/rate limits/invalid
responses, and secret redaction:

```bash
cd pipeline
pytest tests/
```

Expected output:
```text
144 passed, 7 skipped, 5 warnings
```

The 7 skipped tests are the opt-in live integration tests. They are skipped
unless `RUN_LIVE_INTEGRATION=1`, so the ordinary suite makes no external
API calls.

Evaluate the verdict policy on the labelled validation set:

```bash
python scripts/validate_verdict_policy.py
python scripts/validate_verdict_policy.py --sweep
python scripts/validate_verdict_policy.py --legacy   # before/after comparison
```

The containment layer has its own suite:

```bash
pytest verifacto/tests -q
# 94 passed
```

The only pipeline warnings are pre-existing deprecations (`datetime.utcnow`
in the response schema).

### Live integration tests (opt-in, real API calls)

These tests call the real DeepSeek, Tavily, and Google Fact Check APIs. They
consume quota, so they are never part of the default suite. Required
environment variables (values are never printed):

- `LLM_PROVIDER=deepseek`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL` (the current `.env` uses `deepseek-flash`)
- `TAVILY_API_KEY`
- `GOOGLE_FACTCHECK_API_KEY`

Run only the live tests:

```bash
cd pipeline

# Git Bash / macOS / Linux
RUN_LIVE_INTEGRATION=1 pytest tests/test_live_integration.py -m live -s

# PowerShell
$env:RUN_LIVE_INTEGRATION="1"; pytest tests/test_live_integration.py -m live -s
```

The live suite does four things:

1. Calls DeepSeek once with a tiny JSON prompt and verifies parsed output.
2. Calls Tavily once for a common query and verifies real result URLs.
3. Calls Google Fact Check once for a known ClaimReview query and verifies
   returned claims, publishers, and ratings.
4. Runs three full non-blacklisted claims (credible, false, ambiguous)
   through the real pipeline, plus one Tier 4 blacklist case that must make
   zero provider calls.

The live harness resets HTTP clients between pytest event loops and forces a
vector-cache miss so every full-pipeline case actually performs evidence
retrieval and verification. Production behavior is unchanged. Expected
verdicts are recorded and printed, not forced: if live evidence does not
meet the configured confidence/credibility thresholds, the report shows
`expectation_matched=False` instead of altering the detector.

Type checking (optional, uses the repository's `pyrefly.toml`):

```bash
pip install pyrefly
python -m pyrefly check
```

---

## Verification Pipeline

**Step 0 — Tier 4 source gate (optional):** If the caller supplied
`source_url`, normalize the hostname and check the configured blacklist
before any other work. A match returns `FAKE` immediately.

1. **Claim Extraction**: Extracts atomic checkable assertions using Gemini Flash + spaCy NER.
2. **Claim Normalization**: Resolves pronouns, relative dates (e.g. "yesterday" → ISO dates), quantities, and canonical entity names.
3. **Claim Matching**: Checks local vector embeddings and Google ClaimReview for existing verdicts (fast path). A cached verdict is reused only when the claims are materially equivalent: exact normalized text, or matching entities/locations (including entity order), dates, numbers/quantities/currency, and negation with sufficient content-token and content-order overlap. Contradictory claims such as “Louvre in Paris” vs “Louvre in Berlin”, or role reversals such as “Alice gave Bob the book” vs “Bob gave Alice the book”, are re-verified instead of inheriting a cached verdict. Cached verdicts are mapped through the same user-facing policy.
4. **Evidence Retrieval**: Generates 5 diverse search queries and retrieves documents via Tavily API and live web extraction.
5. **Evidence Re-ranking**: Scores retrieved passages with `cross-encoder/ms-marco-MiniLM-L-6-v2` based on relevance, credibility, freshness, and primary source weighting.
6. **Claim Verification**: Performs atomic sub-claim verification and chain-of-thought evidence synthesis with `cross-encoder/nli-deberta-v3-base`.
7. **Confidence Estimation**: Computes a calibrated confidence score (0.00 – 1.00) based on source agreement, contradiction penalties, and evidence quality.
8. **Human Escalation**: Flags high-impact domains (health, war, elections, public disasters) with low confidence for human review.
9. **Verdict Policy**: Maps the internal evidence verdict to exactly one of `REAL`, `FAKE`, or `UNCERTAIN`, applying the evidence-strength gates described above.
10. **Persistence**: Saves claims, evidence, stances, and audit traces to PostgreSQL and Qdrant. The internal four-way verdict is stored; the user-facing verdict is returned.

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
│   ├── pytest.ini                # Registers the opt-in "live" marker
│   ├── config/
│   │   └── blacklisted_sources.json  # Tier 4 publisher-policy registry
│   ├── data/
│   │   └── validation_claims.json    # Labelled policy validation set
│   ├── scripts/
│   │   └── validate_verdict_policy.py  # Policy metrics and threshold sweep
│   ├── .qdrant_storage/          # Persistent local vector database
│   ├── tests/                    # Automated test suite
│   │   ├── test_api.py
│   │   ├── test_blacklist.py     # Tier 4 matching and short-circuit tests
│   │   ├── test_claim_equivalence.py # Material-fact cache guard tests
│   │   ├── test_claim_matcher_fastpath.py # Contradictory-claim fast-path tests
│   │   ├── test_vector_cache_policy.py # Vector payload policy-version tests
│   │   ├── test_confidence.py
│   │   ├── test_deepseek_llm.py
│   │   ├── test_escalation.py
│   │   ├── test_language.py
│   │   ├── test_live_integration.py  # Opt-in real API tests
│   │   ├── test_nli_mapping.py   # NLI label-order regression tests
│   │   ├── test_openai_compatible_llm.py
│   │   ├── test_provider_failures.py # Timeouts, retries, rate limits, redaction
│   │   ├── test_retry_policy.py  # Transient-vs-configuration retry rules
│   │   ├── test_source_credibility_registry.py
│   │   ├── test_verdict_policy.py
│   │   └── test_verdict_validation.py
│   └── src/
│       ├── main.py               # FastAPI application
│       ├── config.py             # Settings & dynamic .env resolution
│       ├── api/routes.py         # HTTP endpoints
│       ├── pipeline/
│       │   ├── orchestrator.py   # Pipeline controller (concurrent execution)
│       │   ├── verdict_policy.py # REAL/FAKE/UNCERTAIN decision policy
│       │   ├── claim_extractor.py
│       │   ├── claim_normalizer.py
│       │   ├── claim_matcher.py
│       │   ├── claim_equivalence.py # Material-fact guard for semantic cache reuse
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
│       │   ├── blacklist.py      # Tier 4 registry loader and matcher
│       │   ├── http_retry.py     # Transient-only retry predicate
│       │   ├── language.py       # Language detection & translation
│       │   ├── redaction.py      # Removes keys/passwords from logs/errors
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
