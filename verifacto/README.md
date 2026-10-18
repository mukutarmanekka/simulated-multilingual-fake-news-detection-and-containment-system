# VeriFacto Containment Layer

The action half of the VeriFacto capstone: a simulated social network where the existing
fake-news detector's verdicts trigger containment decisions, measured against a no-protection
baseline. The detector in `../pipeline/` is treated as a black box and is never modified.

```
detect -> model spread -> decide intervention under a budget -> act -> measure -> feed back
```

## 1. Prerequisites

- Python 3.13 venv at `pipeline/venv` (networkx, numpy, matplotlib, pytest already installed).
- For the real detector only: API keys in the repo-root `.env` (`GEMINI_API_KEY`,
  `TAVILY_API_KEY`, `GOOGLE_FACTCHECK_API_KEY`). The simulation itself needs no keys and no
  running server.
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
| `main blanket` | Detector (stub) + blanket halving for flagged fakes; prints FP cost. | `output/blanket_vs_baseline.png` |
| `main precompute` | Runs the REAL detector over all posts, caches verdicts. Resume-able, paced, guards against mock/degraded verdicts. | `data/detector_cache.json` |
| `main targeted [--detector stub\|replay\|real] [--runs 30] [--k 25] [--allow-live]` | Full comparison: baseline, blanket, random, degree, betweenness, pagerank. Prints metrics + FP cost. | `output/strategy_comparison.png`, `output/fp_cost_bar.png`, `output/metrics.json` |
| `main compare [--detector ...] [--runs 30] [--budgets 10 25] [--greedy-pool 60] [--greedy-runs 5]` | Same but at several budgets, greedy included, one unified table. | `output/strategy_comparison_k{K}.png`, `output/compare_metrics.json` |
| `main sweep p --values 0.05 0.1 0.15` | Vary the spread probability (topology unchanged). | `output/sweep_p.png/json` |
| `main sweep k --values 5 10 25 50` | Vary the budget. | `output/sweep_k.png/json` |
| `main sweep accuracy --values 0.7 0.8 0.9` | Vary stub sensitivity. | `output/sweep_accuracy.png/json` |
| `main bridges` | Louvain sanity check + heavy cross-language connectors. | console |
| `main priority [--detector ...] [--runs 10]` | Ranks flagged posts by confidence x expected reach. | console |
| `main greedy [--detector ...] [--runs 5] [--k 10] [--pool 60]` | Greedy throttling vs betweenness vs baseline (empirical heuristic). | `output/greedy_comparison.png` |

Notes:
- `--detector replay` fails loudly if the cache is missing; run `precompute` first.
- `--detector real --allow-live` calls the pipeline live: ~40 s/post, non-deterministic, so
  never compare live runs against each other. Use it only for spot checks.
- `sweep` excludes greedy unless you pass `--with-greedy` (its selection costs minutes per value).
- Running a command again overwrites its output files; copy them if you need to keep both.

## 4. The real detector: precompute and replay

The real pipeline is slow and non-deterministic, so it is run ONCE and replayed:

```bash
./pipeline/venv/Scripts/python.exe -m verifacto.main precompute   # ~1h for 80 posts
./pipeline/venv/Scripts/python.exe -m verifacto.main targeted --detector replay
```

- The cache stores, per post: raw claims (verdict/confidence/escalation), the mapped
  result, and the **provider + model** that produced it. `meta.providers` has a per-post
  map plus a usage summary.
- Resume: posts already in the cache are skipped. Interrupted runs keep progress
  (atomic write after every post).
- Degraded results are never cached: if the LLM quota fails, the pipeline returns a
  "Pipeline error" placeholder, the precompute prints it, does not cache it, and aborts
  after 3 consecutive failures.
- To re-verify specific posts later, delete their entries from `detector_cache.json`
  and re-run `precompute`.

Current cache scorecard (this corpus): 12 fakes caught, 37 reals passed,
0 false positives, 31 uncertain (28 fakes + 3 reals -> no intervention).

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
- **Verdict mapping:** a post is fake iff any claim is REFUTED with confidence >= 0.5;
  real iff nothing is refuted and some claim is SUPPORTED >= 0.5; otherwise uncertain
  (no intervention, counted separately, never counted as a false positive).
- **Greedy:** selects on its own disjoint seed space (`greedy_seed`), so selection does not
  overfit the evaluation draws. It is an empirical heuristic with no (1-1/e) guarantee;
  the objective uses flagged posts only (no label leakage).

## 6. How to analyze the results yourself

### 6.1 Read the console table

For each strategy: mean +/- std over (fake post, run) pairs of the final fraction of
users reached; peak new infections and its step; containment time (cascade duration in
steps); percent reduction vs baseline. Then the FP-cost block: real posts wrongly flagged,
and users of real-news reach lost per run.

### 6.2 Read the JSON files

- `output/metrics.json` (`targeted`): `meta`, `summary` per strategy
  (`final_fraction_reached`, `peak_spread`, `peak_spread_step`, `time_to_containment`,
  `relative_reduction_vs_baseline`, `interventions`), `fp_cost` per strategy.
- `output/compare_metrics.json` (`compare`): `rows`, one per (budget, strategy), with
  `fake_reach_mean/std`, `real_reach_lost_mean/std`, `posts_acted_on`, `users_throttled`.
- `output/sweep_<mode>.json`: `rankings` (strategy order per swept value) and `rows`.

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

### 6.5 What the two headline results say

- **Stub detector** (assumes a good sensor): blanket is best (78% cut), betweenness is the
  best targeted method at the default budget, greedy beats the centralities when run with
  its own selection seeds.
- **Real detector** (precision-first, 30% coverage): blanket is the only strategy that
  moves the needle (26% cut); targeted throttling lands at 2-4%. The action layer's ceiling
  is set by detector coverage, not by selection quality.

## 7. Calibration and parameter changes

### 7.1 Detector operating point (`verifacto/config.py`)

| Constant | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.5 | Min claim confidence to count as fake/real. Lower = more interventions (more coverage, more false positives). |

Important: the replay backend reads *cached mapped results*, so changing this constant
does NOT change replay runs. Either re-map the cached raw claims (the cache keeps the
original claim list) or delete the cache and re-run `precompute`.

Re-map without new API calls (keeps provider attribution):

```python
import json
from verifacto.config import CACHE_PATH
from verifacto.detector.base import map_job_results

NEW_THRESHOLD = 0.7
cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
for entry in cache["verdicts"].values():
    result = map_job_results(entry["results"], NEW_THRESHOLD, provider=entry["provider"])
    entry["mapped"] = {
        "is_fake": result.is_fake,
        "confidence": result.confidence,
        "verdict_summary": result.verdict_summary,
    }
CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
```

To pick a threshold empirically, sweep it over the cached raw claims and print the
confusion matrix at each value (reuse the scorecard snippet in 6.4 with a loop).

Caveat: the detector's confidence is an evidence-quality score, not a calibrated
probability of correctness. Threshold 0.5 is a policy choice; treat a sweep, not a single
value, as the honest calibration evidence.

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

- Changing detector threshold -> replay cache must be re-mapped (7.1).
- Changing post texts or corpus size -> re-run `precompute` (cache is keyed by post id only).
- Changing graph parameters -> run `main graph`; regenerate posts if block membership or
  node ids changed; detections stay valid (the detector only sees text).
- Changing seeds -> previously saved `output/` files no longer correspond to the new run.

## 8. Switching the LLM provider (Ollama or another)

The detector reads `LLM_PROVIDER` and keys from `.env`. To use Ollama, set
`LLM_PROVIDER=ollama` (and optionally `OLLAMA_MODEL`) and ensure Ollama is serving on
`localhost:11434`. Notes:

- The cache records `provider` and `model` per post, and `meta.providers` summarizes them,
  so mixed-provider caches stay auditable.
- `precompute` skips cached posts. To re-verify a subset under the new provider, delete
  their entries from `detector_cache.json` and re-run.
- If the provider silently falls back to `mock`, the adapter raises and the precompute
  aborts; never let mock verdicts enter a cache.

## 9. Troubleshooting

- **`Event loop is closed` / DB warnings during precompute:** the detector's clients bind
  to one event loop. Use `precompute` (one loop for the whole run), not repeated
  `asyncio.run` calls.
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
  tests/                 85 tests
  data/                  generated graph, posts, detector cache (gitignored)
  output/                charts and metrics (gitignored)
```
