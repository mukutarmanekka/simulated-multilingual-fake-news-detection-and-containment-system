"""Shared constants, paths, and seed derivation for the VeriFacto containment layer.

All random quantities in the project derive from one master seed so every
experiment is reproducible.
"""

from pathlib import Path

import os

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"
DATA_DIR = REPO_ROOT / "verifacto" / "data"
OUTPUT_DIR = REPO_ROOT / "verifacto" / "output"

GRAPH_PATH = DATA_DIR / "graph.graphml"
POSTS_PATH = DATA_DIR / "posts.json"
CACHE_PATH = DATA_DIR / "detector_cache.json"

# ── Seeds ────────────────────────────────────────────────────────────

SEED_MASTER = 20261018
GRAPH_SEED = 11
POSTS_SEED = 22
STUB_SEED = 33


def cascade_seed(post_id: int, run: int) -> int:
    """Deterministic per-(post, run) seed, identical for every strategy."""
    state = np.random.SeedSequence([SEED_MASTER, post_id, run]).generate_state(1)[0]
    return int(state) % (2**32)


# salt that puts the greedy's selection draws in a disjoint seed space
# from the evaluation draws, so the greedy never optimizes on the exact
# randomness it is later scored on
GREEDY_SEED_SALT = 99991


def greedy_seed(post_id: int, run: int) -> int:
    """Selection-space seed for the greedy: disjoint from cascade_seed."""
    state = np.random.SeedSequence(
        [SEED_MASTER, post_id, run, GREEDY_SEED_SALT]
    ).generate_state(1)[0]
    return int(state) % (2**32)


# ── Network ──────────────────────────────────────────────────────────

LANGUAGES = ["en", "hi", "fr"]
BLOCK_SIZES = [250, 150, 100]

P_SPREAD = 0.1   # spread probability on every edge (fixed until Step 5 is done)
P_IN = 0.1       # wiring probability inside a language block
P_OUT = 0.01     # wiring probability across language blocks

HUBS_PER_BLOCK = 3
HUB_DEGREE_TARGET = 60
BRIDGES_PER_BLOCK = 2
BRIDGE_EXTRA_PER_BLOCK = 5

# ── Posts ────────────────────────────────────────────────────────────

N_POSTS = 80
REAL_FRACTION = 0.5

# ── Interventions ────────────────────────────────────────────────────

BLANKET_REDUCTION = 0.5   # flagged fake posts spread at half probability everywhere
TARGETED_REDUCTION = 0.0  # selected users' outgoing edges are fully stopped
K_BUDGET = 25
N_RUNS = 30

CONFIDENCE_THRESHOLD = 0.5  # legacy cache mapping only; real mode preserves the detector verdict

# ── Stub detector ────────────────────────────────────────────────────

STUB_SENSITIVITY = 0.85
STUB_FALSE_POSITIVE_RATE = 0.10

# ── Real detector integration ────────────────────────────────────────

# Bump when the detector's verdict policy changes, so replay cache keys
# from an older policy cannot be mistaken for current verdicts.
DETECTOR_POLICY_VERSION = "verdict_policy_v2"

# "record": store a structured error record and take no containment action.
# "raise": abort the experiment on a detector failure.
DETECTOR_ERROR_POLICY = os.environ.get(
    "VERIFACTO_DETECTOR_ERROR_POLICY", "record"
).strip().lower()

# "no_action": uncertain posts get no intervention.
# "review": uncertain posts get no diffusion change but are counted for review.
UNCERTAIN_POLICY = os.environ.get(
    "VERIFACTO_UNCERTAIN_POLICY", "no_action"
).strip().lower()

DETECTOR_TIMEOUT_SECONDS = float(
    os.environ.get("VERIFACTO_DETECTOR_TIMEOUT", "300")
)
DETECTOR_MAX_ATTEMPTS = int(
    os.environ.get("VERIFACTO_DETECTOR_MAX_ATTEMPTS", "1")
)
DETECTOR_RETRY_DELAY_SECONDS = float(
    os.environ.get("VERIFACTO_DETECTOR_RETRY_DELAY", "2")
)

# Direct real-mode CLI runs detect at most this many posts unless the user
# explicitly passes --max-posts. Precompute is the explicit corpus builder.
DEFAULT_LIVE_MAX_POSTS = int(
    os.environ.get("VERIFACTO_LIVE_MAX_POSTS", "10")
)

# ── Real detector pacing ─────────────────────────────────────────────

PACING_SECONDS = 3.0
