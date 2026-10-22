-- =====================================================================
-- Real-Time Fake News Detection — Database Schema
-- =====================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ── Enums ────────────────────────────────────────────────────────────

CREATE TYPE verdict_type AS ENUM (
    'SUPPORTED',
    'REFUTED',
    'NOT_ENOUGH_EVIDENCE',
    'CONFLICTING'
);

CREATE TYPE stance_type AS ENUM (
    'SUPPORTS',
    'REFUTES',
    'NEUTRAL'
);

CREATE TYPE source_tier AS ENUM (
    'TIER_1',
    'TIER_2',
    'TIER_3',
    'UNKNOWN',
    'BLACKLISTED'
);

CREATE TYPE review_status AS ENUM (
    'PENDING',
    'IN_PROGRESS',
    'COMPLETED',
    'DISMISSED'
);

CREATE TYPE input_type AS ENUM (
    'article',
    'headline',
    'social_media',
    'image_caption',
    'video_transcript',
    'audio_transcript',
    'url'
);

-- ── Sources Registry ─────────────────────────────────────────────────

CREATE TABLE sources (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain          TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    tier            source_tier NOT NULL DEFAULT 'UNKNOWN',
    credibility     REAL NOT NULL DEFAULT 0.40 CHECK (credibility >= 0 AND credibility <= 1),
    total_claims    INTEGER NOT NULL DEFAULT 0,
    accurate_claims INTEGER NOT NULL DEFAULT 0,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sources_domain ON sources(domain);
CREATE INDEX idx_sources_tier ON sources(tier);

-- ── Claims ───────────────────────────────────────────────────────────

CREATE TABLE claims (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_text          TEXT NOT NULL,
    input_type          input_type NOT NULL DEFAULT 'article',
    claim_text          TEXT NOT NULL,
    claim_index         INTEGER NOT NULL DEFAULT 0,
    normalized_text     TEXT,
    entities            JSONB DEFAULT '[]'::jsonb,
    dates               JSONB DEFAULT '[]'::jsonb,
    numbers             JSONB DEFAULT '[]'::jsonb,
    resolved_entities   JSONB DEFAULT '[]'::jsonb,
    resolved_dates      JSONB DEFAULT '[]'::jsonb,
    embedding_stored    BOOLEAN NOT NULL DEFAULT FALSE,
    is_checkable        BOOLEAN NOT NULL DEFAULT TRUE,
    -- Language / Translation metadata
    detected_language   TEXT DEFAULT 'en',
    was_translated      BOOLEAN NOT NULL DEFAULT FALSE,
    original_text       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_claims_created_at ON claims(created_at);
CREATE INDEX idx_claims_entities ON claims USING GIN (entities);
CREATE INDEX idx_claims_normalized ON claims USING GIN (normalized_text gin_trgm_ops);

-- ── Claim Matches (Fast-Path) ────────────────────────────────────────

CREATE TABLE claim_matches (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id            UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    matched_claim_id    UUID REFERENCES claims(id),
    match_source        TEXT NOT NULL,  -- 'vector_db', 'exact', 'factcheck_api'
    similarity_score    REAL NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    matched_verdict     verdict_type,
    matched_claim_text  TEXT,
    external_url        TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_claim_matches_claim ON claim_matches(claim_id);

-- ── Evidence Documents ───────────────────────────────────────────────

CREATE TABLE evidence (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id            UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    title               TEXT NOT NULL,
    url                 TEXT NOT NULL,
    source_domain       TEXT,
    source_id           UUID REFERENCES sources(id),
    published_date      TIMESTAMPTZ,
    credibility_score   REAL DEFAULT 0.40 CHECK (credibility_score >= 0 AND credibility_score <= 1),
    relevance_score     REAL DEFAULT 0.0,
    freshness_score     REAL DEFAULT 0.0,
    final_rank_score    REAL DEFAULT 0.0,
    passages            JSONB DEFAULT '[]'::jsonb,
    stance              stance_type,
    is_primary_source   BOOLEAN DEFAULT FALSE,
    retrieval_query     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_evidence_claim ON evidence(claim_id);
CREATE INDEX idx_evidence_rank ON evidence(claim_id, final_rank_score DESC);

-- ── Verification Results ─────────────────────────────────────────────

CREATE TABLE verification_results (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id            UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verdict             verdict_type NOT NULL,
    confidence          REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning           JSONB NOT NULL DEFAULT '[]'::jsonb,
    subclaim_results    JSONB DEFAULT '[]'::jsonb,
    matched_existing    BOOLEAN NOT NULL DEFAULT FALSE,
    matched_claim_id    UUID,
    pipeline_duration_ms INTEGER,
    model_used          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_verification_claim ON verification_results(claim_id);
CREATE INDEX idx_verification_verdict ON verification_results(verdict);
CREATE INDEX idx_verification_confidence ON verification_results(confidence);
CREATE INDEX idx_verification_created ON verification_results(created_at);

-- ── Human Review Queue ───────────────────────────────────────────────

CREATE TABLE human_reviews (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id            UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verification_id     UUID NOT NULL REFERENCES verification_results(id) ON DELETE CASCADE,
    status              review_status NOT NULL DEFAULT 'PENDING',
    escalation_reason   TEXT NOT NULL,
    priority            INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    assigned_to         TEXT,
    reviewer_verdict    verdict_type,
    reviewer_notes      TEXT,
    reviewer_confidence REAL CHECK (reviewer_confidence >= 0 AND reviewer_confidence <= 1),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at         TIMESTAMPTZ
);

CREATE INDEX idx_human_reviews_status ON human_reviews(status);
CREATE INDEX idx_human_reviews_priority ON human_reviews(priority DESC);

-- ── Audit Log ────────────────────────────────────────────────────────

CREATE TABLE audit_log (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    claim_id            UUID REFERENCES claims(id) ON DELETE SET NULL,
    step_name           TEXT NOT NULL,
    step_index          INTEGER NOT NULL,
    input_data          JSONB,
    output_data         JSONB,
    duration_ms         INTEGER,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_claim ON audit_log(claim_id);
CREATE INDEX idx_audit_step ON audit_log(step_name);

-- ── Verification Jobs ────────────────────────────────────────────────

CREATE TABLE verification_jobs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_text          TEXT NOT NULL,
    input_type          input_type NOT NULL DEFAULT 'article',
    status              TEXT NOT NULL DEFAULT 'queued',
    total_claims        INTEGER DEFAULT 0,
    completed_claims    INTEGER DEFAULT 0,
    results             JSONB DEFAULT '[]'::jsonb,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ
);

CREATE INDEX idx_jobs_status ON verification_jobs(status);
CREATE INDEX idx_jobs_created ON verification_jobs(created_at);

-- ── Seed: Tier-1 and Tier-2 Source Registry ──────────────────────────

INSERT INTO sources (domain, name, tier, credibility, description) VALUES
    -- Tier 1: Government & International Organizations
    ('who.int',             'World Health Organization',    'TIER_1', 0.95, 'UN health agency'),
    ('nasa.gov',            'NASA',                         'TIER_1', 0.96, 'US space agency'),
    ('cdc.gov',             'CDC',                          'TIER_1', 0.95, 'US disease control'),
    ('nih.gov',             'NIH',                          'TIER_1', 0.95, 'US health research'),
    ('whitehouse.gov',      'The White House',              'TIER_1', 0.90, 'US executive branch'),
    ('europa.eu',           'European Union',               'TIER_1', 0.92, 'EU official portal'),
    ('un.org',              'United Nations',               'TIER_1', 0.93, 'UN official site'),
    ('gov.uk',              'UK Government',                'TIER_1', 0.92, 'UK official portal'),
    -- Tier 1: Major Wire Services
    ('reuters.com',         'Reuters',                      'TIER_1', 0.94, 'Global wire service'),
    ('apnews.com',          'Associated Press',             'TIER_1', 0.94, 'Global wire service'),
    ('afp.com',             'AFP',                          'TIER_1', 0.93, 'French wire service'),
    -- Tier 2: Quality Journalism
    ('nytimes.com',         'The New York Times',           'TIER_2', 0.88, 'US newspaper of record'),
    ('washingtonpost.com',  'The Washington Post',          'TIER_2', 0.87, 'US major newspaper'),
    ('bbc.com',             'BBC',                          'TIER_2', 0.89, 'UK public broadcaster'),
    ('bbc.co.uk',           'BBC UK',                       'TIER_2', 0.89, 'UK public broadcaster'),
    ('theguardian.com',     'The Guardian',                 'TIER_2', 0.86, 'UK newspaper'),
    ('economist.com',       'The Economist',                'TIER_2', 0.88, 'UK economics magazine'),
    ('nature.com',          'Nature',                       'TIER_2', 0.95, 'Scientific journal'),
    ('science.org',         'Science',                      'TIER_2', 0.95, 'Scientific journal'),
    ('thelancet.com',       'The Lancet',                   'TIER_2', 0.94, 'Medical journal'),
    -- Tier 2: Fact-Checkers
    ('snopes.com',          'Snopes',                       'TIER_2', 0.88, 'Fact-checking site'),
    ('politifact.com',      'PolitiFact',                   'TIER_2', 0.87, 'Political fact-checker'),
    ('factcheck.org',       'FactCheck.org',                'TIER_2', 0.88, 'Non-partisan fact-checker'),
    ('fullfact.org',        'Full Fact',                    'TIER_2', 0.87, 'UK fact-checker'),
    -- Tier 1: Official publisher for a specific entity
    ('toureiffel.paris',    'Eiffel Tower Official Site',    'TIER_1', 0.90, 'Official Eiffel Tower operator publisher'),
    -- Tier 2: Reference / educational
    ('britannica.com',      'Encyclopaedia Britannica',      'TIER_2', 0.85, 'Edited general reference'),
    -- Tier 3: Reference, educational, and user-generated
    ('wikipedia.org',       'Wikipedia',                     'TIER_3', 0.65, 'Crowd-edited reference; moderate prior, corroboration required'),
    ('history.com',         'History.com',                   'TIER_3', 0.72, 'Established history publisher'),
    ('hnn.us',              'History News Network',          'TIER_3', 0.65, 'History commentary and news publisher'),
    ('artsandculture.google.com', 'Google Arts & Culture',   'TIER_3', 0.70, 'Curated cultural platform'),
    ('history.stackexchange.com', 'History Stack Exchange',  'TIER_3', 0.45, 'User-generated Q&A; below evidence gate'),
    ('youtube.com',         'YouTube',                       'TIER_3', 0.35, 'User-generated video platform'),
    ('youtu.be',            'YouTube Short Link',            'TIER_3', 0.35, 'User-generated video platform')
ON CONFLICT (domain) DO NOTHING;
