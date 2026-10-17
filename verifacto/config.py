"""Shared constants, paths, and seed derivation for the VeriFacto containment layer.

All random quantities in the project derive from one master seed so every
experiment is reproducible.
"""

from pathlib import Path

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

CONFIDENCE_THRESHOLD = 0.5

# ── Stub detector ────────────────────────────────────────────────────

STUB_SENSITIVITY = 0.85
STUB_FALSE_POSITIVE_RATE = 0.10

# ── Real detector pacing ─────────────────────────────────────────────

PACING_SECONDS = 3.0
