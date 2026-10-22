# VeriFacto Containment Layer

The action half of the VeriFacto capstone: a simulated social network where the existing
fake-news detector's verdicts trigger containment decisions, measured against a no-protection
baseline. The detector in `../pipeline/` is treated as a black box and is never modified.

For a copy-paste end-to-end runbook (offline stub loop, real-cache replay, live rebuild,
and optional analyses), see [`RUNBOOK.md`](RUNBOOK.md).

```
detect -> model spread -> decide intervention under a budget -> act -> measure -> feed back
```

## 1. Prerequisites

- Python 3.13 venv at `pipeline/venv` (networkx, numpy, matplotlib, pytest already installed).
- The real detector is the actual `pipeline/` implementation. It supports DeepSeek,
  Gemini, OpenAI, OpenAI-compatible, and Ollama; configure one in the repo-root `.env`
  (the current environment uses `LLM_PROVIDER=deepseek`, `DEEPSEEK_API_KEY`,
  `TAVILY_API_KEY`, and `GOOGLE_FACTCHECK_API_KEY`). Never commit key values.
- The simulation itself needs no keys for stub/replay mode and no running server.
- Run everything from the repository root (the folder containing `verifacto/`).

All commands below use `./pipeline/venv/Scripts/python.exe` (Git Bash on Windows). In
PowerShell use `.\pipeline\venv\Scripts\python.exe`.

## 2. Quick start (the full loop, ~4 minutes with the stub detector)

```bash
# 1. build the 500-user language-block graph (en/hi/fr) and save it
./pipeline/venv/Scripts/python.exe -m verifacto.main graph

# 2. generate the 80-post corpus (40 real / 40 fake, hidden labels)
./pipeline/venv/Scripts/python.exe -m verifacto.main posts

# 3. Independent Cascade with no protection (baseline)
./pipeline/venv/Scripts/python.exe -m verifacto.main baseline

# 4. detector + blanket halving of flagged fakes
./pipeline/venv/Scripts/python.exe -m verifacto.main blanket

# 5. targeted top-k containment vs random, with all metrics
./pipeline/venv/Scripts/python.exe -m verifacto.main targeted --detector stub --runs 30 --k 25
```

Each command prints its results and writes charts/metrics to `verifacto/output/`.
The detector defaults to the deterministic stub; use `--detector replay` to run on the
cached real-detector verdicts (see section 4).

Run the tests at any time:

```bash
./pipeline/venv/Scripts/python.exe -m pytest verifacto/tests -q
```

## 3. Command reference

| Command | What it does | Writes |
|---|---|---|
| `main graph` | Builds the SBM graph: en 250 / hi 150 / fr 100, hubs, bilingual bridges, p=0.1 per edge. Prints block sizes, cross-language edge fraction, densities. | `data/graph.graphml` |
| `main posts` | Generates 80 posts from hand-written templates per language; authors in the post's block. | `data/posts.json` |
| `main baseline` | IC with no protection, `N_RUNS` seeded runs per post. | `output/baseline_reach.png`, `output/baseline_series.json` |
| `main blanket [--detector stub\|replay\|real] [--allow-live] [--max-posts N]` | Detector + blanket halving for flagged fakes; prints detector quality and FP cost. | `output/blanket_vs_baseline.png`, `output/blanket_metrics.json` |
| `main precompute [--max-posts N] [--posts-path PATH] [--pacing S] [--recompute-stale]` | Runs the REAL detector and caches full verdict/evidence records. Resume-able, paced, guards against mock/degraded verdicts. | `data/detector_cache.json` |
| `main targeted [--detector stub\|replay\|real] [--runs 30] [--k 25] [--allow-live] [--max-posts N] [--detector-error-policy record\|raise]` | Full comparison: baseline, blanket, random, degree, betweenness, pagerank. Prints detector summary, metrics, FP cost. | `output/strategy_comparison.png`, `output/fp_cost_bar.png`, `output/metrics.json`, `output/metrics_<mode>.json` |
| `main compare [--detector ...] [--runs 30] [--budgets 10 25] [--greedy-pool 60] [--greedy-runs 5] [--allow-live] [--max-posts N]` | Same but at several budgets, greedy included, one unified table. | `output/strategy_comparison_k{K}.png`, `output/compare_metrics.json`, mode copy |
| `main sweep p --values 0.05 0.1 0.15` | Vary the spread probability (topology unchanged). | `output/sweep_p.png/json` |
| `main sweep k --values 5 10 25 50` | Vary the budget. | `output/sweep_k.png/json` |
| `main sweep accuracy --values 0.7 0.8 0.9` | Vary stub sensitivity. | `output/sweep_accuracy.png/json` |
| `main bridges` | Louvain sanity check + heavy cross-language connectors. | console |
| `main priority [--detector ...] [--runs 10] [--allow-live] [--max-posts N]` | Ranks flagged posts by confidence x expected reach. | `output/priority.json`, mode copy |
| `main greedy [--detector ...] [--runs 5] [--k 10] [--pool 60] [--allow-live] [--max-posts N]` | Greedy throttling vs betweenness vs baseline (empirical heuristic). | `output/greedy_comparison.png`, `output/greedy_metrics.json` |

Notes:
- `--detector replay` fails loudly if the cache is missing or a cache key does not match
  the current claim/language/source; run `precompute` first.
- `--detector real --allow-live` calls the pipeline live: ~40 s/post, non-deterministic, so
  never compare live runs against each other. Use it only for spot checks. Direct real-mode
  commands process at most `--max-posts` (default 10) unless you set it explicitly.
- `sweep` excludes greedy unless you pass `--with-greedy` (its selection costs minutes per value).
- Commands that accept a detector write both the canonical output (e.g. `metrics.json`) and
  a mode-specific copy (`metrics_stub.json`, `metrics_replay.json`, or `metrics_real.json`).
  The canonical file includes `meta.detector_mode`; the mode-specific copy is the safer
  artifact to keep when comparing experiments.
- `--detector-error-policy record` (default) records a `status="error"`/`"timeout"` detection
  and applies no containment action. `raise` aborts the experiment. A mock LLM provider or
  an import failure always aborts; it is never replaced by the stub.
- `VERIFACTO_UNCERTAIN_POLICY=no_action` (default) gives uncertain posts no intervention.
  `review` keeps the same diffusion behavior but reports the uncertain review queue. There
  is no automatic suppression of uncertain content.
- `VERIFACTO_DETECTOR_TIMEOUT` (default 300 s), `VERIFACTO_LIVE_MAX_POSTS` (default 10),
  and `VERIFACTO_DETECTOR_MAX_ATTEMPTS` (default 1) control live-call safety.

## 4. The real detector: real, precompute, and replay

### Real mode

`RealDetectorAdapter` is the only module that imports the detector. It calls the
detector's actual public entry point:

```text
PipelineOrchestrator.run(input_text, input_type=social_media,
                         source_url=<optional post source>)
```

It preserves `REAL`/`FAKE`/`UNCERTAIN`, confidence, evidence support score, reason code,
evidence documents, source credibility/relevance, Tier 4 metadata, latency, and
provider-call counts. It does not reimplement verification policy and does not apply a
second confidence threshold.

```bash
# bounded live run (3 posts, 2 cascade runs, k=5)
./pipeline/venv/Scripts/python.exe -m verifacto.main targeted \
  --detector real --allow-live --max-posts 3 --runs 2 --k 5
```

Direct real-mode commands default to `--max-posts 10`. The real pipeline may use its
own internal vector/fact-check fast path; the detection record exposes
`matched_existing_claim` and the reason. That fast path is material-fact guarded: a
cached verdict is reused only for an exact or materially equivalent claim (matching
entities/locations and their order, dates, numbers/quantities/currency, and negation
with sufficient content-token and content-order overlap), and the detector's vector
payload is policy-versioned. Lexical substitutions are default-deny: a changed content
word is re-verified unless it is a known inflection or paraphrase, so “increased” vs
“decreased”, “safe” vs “unsafe”, and place-name swaps in uncased scripts such as Hindi
cannot inherit a cached verdict. A contradictory claim
such as “Louvre in Paris” vs “Louvre in Berlin” is also re-verified instead of inheriting
the cached verdict. This is distinct from the simulation's replay mode, which uses only
`detector_cache.json` and enforces its own claim/language/source/policy cache key.

### Precompute and replay

The full corpus is slow and non-deterministic, so it is normally precomputed once and
replayed:

```bash
./pipeline/venv/Scripts/python.exe -m verifacto.main precompute
./pipeline/venv/Scripts/python.exe -m verifacto.main targeted \
  --detector replay --cache-provider deepseek --cache-model deepseek-flash

# bounded precompute; current entries are skipped
./pipeline/venv/Scripts/python.exe -m verifacto.main precompute --max-posts 5
./pipeline/venv/Scripts/python.exe -m verifacto.main precompute \
  --max-posts 15 --recompute-stale
# custom corpus with source URLs
./pipeline/venv/Scripts/python.exe -m verifacto.main precompute --posts-path my_posts.json
```

- Cache entries are keyed to the claim text, language, source URL, and detector policy
  version. Replay raises `DetectorCacheMiss` on a key mismatch instead of attaching a
  stale verdict to a different claim.
- Replay can require a specific provider/model with `--cache-provider` and `--cache-model`.
  Precompute also treats a policy-version, provider, or model mismatch as stale, so
  `precompute --recompute-stale` rebuilds exactly those entries and skips current ones.
  The entry stores provider/model for attribution.
- Each entry stores a full `record` (verdict, confidence, support score, reason, evidence,
  blacklist metadata, status, latency, provider usage), the legacy `mapped` fields, raw
  claim metadata, provider, model, and cache key.
- `--recompute-stale` rebuilds entries whose policy version is missing or stale and skips
  current entries, so cumulative `--max-posts` batches do not repeat verified work. A
  timestamped backup (`data/detector_cache.backup-*.json`) is written before a rebuild.
- Degraded results are never cached: if the LLM quota fails, the pipeline returns a
  "Pipeline error" placeholder, the precompute prints it, does not cache it, and aborts
  after 3 consecutive failures.
- To re-verify specific posts later, use `--recompute-stale` (or delete their entries).

The current cache has been migrated: **80/80 entries are current** (`verdict_policy_v2`,
DeepSeek/`deepseek-flash`, cache keys and full records). The original pre-migration cache
is preserved in a timestamped backup. Some fast-path entries have no evidence documents
because the detector reused a materially equivalent cached claim; their records show
`matched_existing_claim` and the reason.

### Tier 4 blacklist

Tier 4 is enforced by the detector, not the simulation. The adapter passes the post's
optional `source_url`; a configured satire/parody/troll/fake-news source returns the
detector's immediate `FAKE` with `BLACKLISTED_<CATEGORY>_SOURCE` and zero downstream
provider calls. The simulation preserves the blacklist reason/domain/category and treats
the post as FAKE for containment. It never does its own substring matching or treats
unknown/low-credibility domains as fake.

### Bounded live test

```bash
RUN_VERIFACTO_LIVE=1 ./pipeline/venv/Scripts/python.exe -m pytest \
  verifacto/tests/test_live_detector_integration.py -s -q
```

This makes at most three detections: two non-blacklisted claims and one The Onion post.
It reports actual verdicts, latency, provider usage, and the resulting tiny containment
experiment. The satire case must make zero provider calls. If credentials are unavailable,
the test skips and live integration is NOT TESTED.

## 5. How the simulation works (so you can reason about changes)

- **Graph:** `nx.random_partition_graph(..., p_in=0.1, p_out=0.01, directed=True)` builds the
  blocks; hubs get same-block outgoing edges until out-degree 60; bridges get extra
  cross-block edges. Every edge stores spread probability `p=0.1`.
- **Cascade (D3):** for each (post, run), one seeded RNG builds a draw table
  `draws[(u,v)] = rng.random()` over sorted edges. An edge succeeds iff
  `draws[(u,v)] < p_effective(u,v)` and every strategy shares the same table. Since
  interventions only lower probabilities, intervention reach is an exact subset of
  baseline reach per (post, run). Never replace this with an attempt-order RNG.
- **Interventions:** probability maps, never graph mutation.
  `p_eff = p * post_factor * user_factor(u)`. Centralities are computed on the original graph.
- **Verdict mapping:** the real adapter preserves the detector's public
  `REAL` / `FAKE` / `UNCERTAIN` verdict. `is_fake` is True only for FAKE, False for REAL,
  and None for UNCERTAIN. Execution failures (`status="error"`/`"timeout"`) have
  `verdict=None` and are never treated as FAKE or as a genuine UNCERTAIN result.
  `UNCERTAIN` and `NO_CLAIMS` get no containment action by default; they are counted
  separately from false positives and false negatives. Legacy `REFUTED`/`SUPPORTED`
  cache values are mapped to FAKE/REAL for compatibility.
- **Greedy:** selects on its own disjoint seed space (`greedy_seed`), so selection does not
  overfit the evaluation draws. It is an empirical heuristic with no (1-1/e) guarantee;
  the objective uses flagged posts only (no label leakage).

### Detector quality vs containment effectiveness

`detector_summary` is computed once per experiment and is shared by every strategy.
Strategy comparisons reuse the same posts, ground truth, detector verdicts, and cascade
seeds; only the containment probability maps differ. Do not compare strategies that used
different detector modes or different post subsets. The `detections` section is the
authoritative per-post record for reconstructing a run.

The real detector receives only the post text, language, post id, and optional source
URL. Synthetic `true_label` values are used only in `evaluation/metrics.py` after
detection; they never influence the detector or containment selection.

`REAL` posts are not suppressed unless a strategy deliberately restricts all content
(the current blanket strategy only halves flagged FAKE posts). `UNCERTAIN` and
`NO_CLAIMS` posts receive no containment by default and are counted separately.
`status="error"`/`"timeout"` detection failures are counted separately and never
masquerade as a verdict.

### Synthetic results

The network, posts, cascades, and reach curves are a simulation. They are not observed
real-world misinformation spread. API usage and latency in the detection records are
actual measurements when real mode is used; synthetic ground truth is not.

## 6. How to analyze the results yourself

### 6.1 Read the console table

For each strategy: mean +/- std over (fake post, run) pairs of the final fraction of
users reached; peak new infections and its step; containment time (cascade duration in
steps); percent reduction vs baseline. Then the FP-cost block: real posts wrongly flagged,
and users of real-news reach lost per run.

### 6.2 Read the JSON files

- `output/metrics.json` (`targeted`): `meta` (`detector`, `detector_mode`,
  `detector_error_policy`, `uncertainty_policy`, `n_posts`, `n_runs`, `k`),
  `detector_summary` (verdict counts, status counts, uncertainty, errors, blacklist
  count, mean confidence/support/latency, provider usage, ground-truth confusion,
  false-positive/dangerous-false-negative rates), `detections` (one full record per
  post), `summary` per strategy (`final_fraction_reached`, `peak_spread`,
  `peak_spread_step`, `time_to_containment`, `relative_reduction_vs_baseline`,
  `interventions`), and `fp_cost` per strategy. The same payload is copied to
  `metrics_<mode>.json`.
- `output/compare_metrics.json` (`compare`): `meta`, `detector_summary`, `detections`,
  and `rows`, one per (budget, strategy), with `fake_reach_mean/std`,
  `real_reach_lost_mean/std`, `posts_acted_on`, `users_throttled`. A mode-specific copy
  is written.
- `output/blanket_metrics.json`, `output/greedy_metrics.json`,
  `output/priority.json`: the same detector summary/detection records plus their
  strategy-specific tables. Mode-specific copies are written.
- `output/sweep_<mode>.json`: `rankings` (strategy order per swept value) and `rows`;
  sweeps are stub-only and identify `detector_mode="stub"`.
- Every detection record includes the post id, original claim/text, language, source
  URL/domain, verdict, confidence, support score, reason, evidence documents, source
  credibility/relevance, blacklist fields, status/error, latency, and provider usage.

### 6.3 Sanity checks (what makes the numbers trustworthy)

1. **Dominance:** for any fixed (post, run), protected reach <= baseline reach at every
   step. The test suite asserts this; if it ever fails, the draw-table contract broke.
2. **Same seeds:** all strategies for a given (post, run) must use
   `cascade_seed(post_id, run)`. This is why differences are attributable to the strategy.
3. **Uncertain is not FP:** uncertain detections get no intervention and are reported
   separately. Only real posts flagged fake count as false positives.
4. **Determinism:** rerun the same command; `metrics.json` and the PNGs are byte-identical
   (verified for `targeted --detector replay`).
5. **Per-post heterogeneity** drives the large std: some authors sit in dense areas, some
   on the edge. For cleaner comparisons, use paired differences per (post, run):

```python
import numpy as np
# results: {post_id: [PostRunResult]} for two strategies from the same experiment
diffs = []
for post_id in baseline_results:
    for a, b in zip(baseline_results[post_id], strategy_results[post_id]):
        diffs.append(a.total_reached - b.total_reached)
mean = np.mean(diffs)
se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
print(f"mean reduction {mean:.2f} users, 95% CI [{mean - 1.96*se:.2f}, {mean + 1.96*se:.2f}]")
```

### 6.4 Score any detector backend against the hidden labels

Only this analysis may read `true_label` (metrics-only rule):

```python
import json
from verifacto.data.posts import load_posts
from verifacto.config import CACHE_PATH

posts = load_posts()
cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
tp = tn = fp = fn = uncertain = 0
for post in posts:
    verdict = cache["verdicts"][str(post.id)]["mapped"]["is_fake"]
    if verdict is None:
        uncertain += 1
    elif verdict is True and post.true_label == "fake":
        tp += 1
    elif verdict is False and post.true_label == "real":
        tn += 1
    elif verdict is True:
        fp += 1
    else:
        fn += 1
print(f"TP={tp} TN={tn} FP={fp} FN={fn} uncertain={uncertain}")
```

### 6.5 Historical headline numbers

Numbers in older notes or output files were produced by an earlier detector policy and
the old cache. They are historical. The current `detector_cache.json` has been rebuilt
under the current policy and replay verifies its keys; rerun the comparison commands to
regenerate current outputs.

## 7. Calibration and parameter changes

### 7.1 Detector operating point

Real mode uses the detector's own `REAL`/`FAKE`/`UNCERTAIN` policy. The simulation does
not apply a second confidence threshold to real verdicts. `CONFIDENCE_THRESHOLD` remains
only for the legacy `map_job_results` path used by old cache entries and unit tests.

To refresh old cache entries with the current detector policy:

```bash
./pipeline/venv/Scripts/python.exe -m verifacto.main precompute --recompute-stale
```

This makes live calls and rewrites the selected entries with full records and cache keys.
To inspect or calibrate the detector itself, use its own labelled validation set
(`pipeline/data/validation_claims.json`) and `pipeline/scripts/validate_verdict_policy.py`;
do not tune the detector from the simulation's synthetic outcomes.

### 7.2 Stub detector (simulated sensor)

| Constant | Default | Effect |
|---|---|---|
| `STUB_SENSITIVITY` | 0.85 | P(flag | truly fake) |
| `STUB_FALSE_POSITIVE_RATE` | 0.10 | P(flag | truly real) |
| `STUB_SEED` | 33 | Fixes which specific posts are misclassified |

Or from code: `StubDetector(posts, sensitivity=0.9, false_positive_rate=0.05)`.
`sweep accuracy` varies sensitivity; FPR stays at its config value.

### 7.3 Spread and intervention parameters

| Constant | Default | Notes |
|---|---|---|
| `P_SPREAD` | 0.1 | Spread probability on every edge. Direct edits apply only after `main graph` rebuilds the file. `sweep p` varies it in memory. |
| `P_IN` / `P_OUT` | 0.1 / 0.01 | Wiring density. Changing these changes TOPOLOGY -> rebuild the graph and regenerate posts. |
| `BLANKET_REDUCTION` | 0.5 | Multiplier on all edges of a flagged post. |
| `TARGETED_REDUCTION` | 0.0 | Multiplier on outgoing edges of throttled users (0 = full stop). |
| `K_BUDGET` | 25 | Default number of throttled users. |
| `N_RUNS` | 30 | Monte Carlo runs per (post, strategy). |
| `HUBS_PER_BLOCK` etc. | 3 / 60 / 2 / 5 | Hub/bridge construction. Changing requires a graph rebuild. |

### 7.4 Seeds and reproducibility

| Constant | Controls |
|---|---|
| `SEED_MASTER` | All cascade draws (via `cascade_seed`). Changes results, not topology/posts. |
| `GRAPH_SEED` | Graph wiring. Requires `main graph` + `main posts`. |
| `POSTS_SEED` | Which templates/placeholders/authors are chosen. Requires `main posts`. |
| `GREEDY_SEED_SALT` | Keeps greedy selection draws disjoint from evaluation draws. Do not set it equal to the evaluation space. |

### 7.5 What invalidates what

- Changing the detector policy or provider -> rerun `precompute --recompute-stale`; new
  cache keys include the policy version, claim, language, and source URL.
- Changing post texts or corpus size -> rerun `precompute`; the cache key binds the claim
  text, language, source URL, and policy version. The post id is the lookup key, and a
  mismatch raises `DetectorCacheMiss`.
- Changing graph parameters -> run `main graph`; regenerate posts if block membership or
  node ids changed; detections stay valid (the detector only sees text).
- Changing seeds -> previously saved `output/` files no longer correspond to the new run.

## 8. Switching the LLM provider (Ollama or another)

The detector reads `LLM_PROVIDER` and keys from `.env`. To use Ollama, set
`LLM_PROVIDER=ollama` (and optionally `OLLAMA_MODEL`) and ensure Ollama is serving on
`localhost:11434`. Notes:

- The cache records `provider` and `model` per post, and `meta.providers` summarizes them,
  so mixed-provider caches stay auditable.
- `precompute` skips cached posts. To re-verify a subset under the new provider, use
  `--recompute-stale` (or delete the entries) and re-run.
- If the provider silently falls back to `mock`, the adapter raises and the precompute
  aborts; never let mock verdicts enter a cache.

## 9. Troubleshooting

- **`Event loop is closed` / DB warnings:** the detector's clients bind to one event loop.
  The simulation's `run_detection` awaits all real detections in one loop and `precompute`
  does the same. Do not call the adapter's synchronous `detect()` in a loop; use
  `run_detection` or `detect_async` inside one loop.
- **Gemini 429 / quota exhausted:** the free tier is 20 requests/day. Degraded verdicts are
  not cached; fix the key or switch provider and resume.
- **`UnicodeEncodeError` in the console:** set `PYTHONIOENCODING=utf-8`.
- **`DetectorCacheMiss`:** the cache lacks that post; run `precompute` or restore the cache.
- **Graph file stale after config edits:** `targeted`/`compare` load `data/graph.graphml`;
  run `main graph` after changing wiring parameters.
- **Empty legend/labels in charts:** matplotlib uses the Agg backend and the validated
  palette; charts are PNGs under `output/`.

## 10. Layout

```
verifacto/
  config.py              constants, paths, seed derivation
  main.py                CLI driver (all subcommands)
  network/graph_builder.py
  data/posts.py
  detector/{base,stub,real_adapter,replay,precompute}.py
  simulation/{cascade,intervention,experiment}.py
  containment/{heuristics,greedy,bridges,priority}.py
  evaluation/{metrics,plots}.py
  tests/                 131 offline tests + 1 opt-in live test
  data/                  generated graph, posts, detector cache (gitignored)
  output/                charts and metrics (gitignored)
```
