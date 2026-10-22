# Running VeriFacto End to End

This is a self-contained runbook for the VeriFacto containment layer. It covers the
offline stub loop, replaying the cached real-detector verdicts, rebuilding the real cache,
and the optional Step 5/6 analyses.

Run every command from the repository root (the folder containing `verifacto/`).

```bash
cd "E:/Codes/Projects/RealTimeFakeNewsDetection - Copy"
export PYTHONIOENCODING=utf-8
PY=./pipeline/venv/Scripts/python.exe
```

PowerShell equivalent:

```powershell
cd "E:\Codes\Projects\RealTimeFakeNewsDetection - Copy"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\pipeline\venv\Scripts\python.exe"
```

## 1. Prerequisites

- Python 3.13 virtual environment at `pipeline/venv`.
- The simulation needs **no API keys and no running server** in stub or replay mode.
- Live real-detector mode needs a configured repo-root `.env`:
  - `LLM_PROVIDER=deepseek`
  - `DEEPSEEK_API_KEY`
  - `TAVILY_API_KEY`
  - `GOOGLE_FACTCHECK_API_KEY`

Never commit `.env`, cache files, or cache backups.

If the venv has to be recreated:

```bash
python -m venv pipeline/venv
./pipeline/venv/Scripts/python.exe -m pip install -r pipeline/requirements.txt
./pipeline/venv/Scripts/python.exe -m pip install -r verifacto/requirements.txt
```

Optional sanity check before running experiments:

```bash
$PY -m pytest verifacto/tests -q
```

Relevant detector/cache tests:

```bash
cd pipeline
../pipeline/venv/Scripts/python.exe -m pytest \
  tests/test_claim_equivalence.py \
  tests/test_claim_matcher_fastpath.py \
  tests/test_vector_cache_policy.py \
  tests/test_verdict_policy.py \
  tests/test_verdict_validation.py \
  tests/test_escalation.py \
  tests/test_confidence.py -q
cd ..
```

## 2. Detector modes

| Mode | Verdict source | API calls | Deterministic |
|---|---|---|---|
| `stub` | Synthetic detector with known sensitivity/FP rate | No | Yes |
| `replay` | Cached real-detector verdicts in `data/detector_cache.json` | No | Yes |
| `real` | Live `pipeline/` detector | Yes | No |

Use `stub` for fast development, `replay` for the real-detector results, and `real` only
for bounded spot checks or a deliberate cache rebuild. Never compare one live run against
another live run.

## 3. Track A — full loop offline (recommended first run)

This runs the complete loop with the deterministic stub detector. It takes a few minutes.

```bash
# 1. Build the 500-user language-block graph (en 250 / hi 150 / fr 100)
$PY -m verifacto.main graph

# 2. Generate the 80-post corpus (40 real / 40 fake, hidden labels)
$PY -m verifacto.main posts

# 3. Unprotected Independent Cascade baseline
$PY -m verifacto.main baseline

# 4. Detector + blanket halving of flagged fake posts
$PY -m verifacto.main blanket

# 5. Targeted top-k containment vs random, degree, betweenness, and PageRank
$PY -m verifacto.main targeted --detector stub --runs 30 --k 25
```

What to look for:

- `graph` prints block sizes, cross-language edge fraction, and densities.
- `posts` prints 40 real / 40 fake posts with authors in the matching language.
- `baseline` produces the unprotected “users reached over time” curve.
- `blanket` prints detector quality and false-positive cost; the protected curve must be
  lower than the baseline.
- `targeted` prints the strategy comparison table. Betweenness is normally the best
  targeted heuristic.

Artifacts:

| Command | Files written under `verifacto/output/` |
|---|---|
| `baseline` | `baseline_reach.png`, `baseline_series.json` |
| `blanket` | `blanket_vs_baseline.png`, `blanket_metrics.json`, `blanket_metrics_stub.json` |
| `targeted` | `strategy_comparison.png`, `fp_cost_bar.png`, `metrics.json`, `metrics_stub.json` |

## 4. Track B — real detector verdicts without live calls

The cache currently has 80/80 current entries (`verdict_policy_v2`, DeepSeek /
`deepseek-flash`). Replay uses those verdicts and makes zero API calls.

```bash
$PY -m verifacto.main targeted --detector replay --runs 30 --k 25 \
  --cache-provider deepseek --cache-model deepseek-flash

$PY -m verifacto.main blanket --detector replay \
  --cache-provider deepseek --cache-model deepseek-flash

$PY -m verifacto.main compare --detector replay --runs 30 --budgets 10 25 \
  --cache-provider deepseek --cache-model deepseek-flash

$PY -m verifacto.main priority --detector replay --runs 10 \
  --cache-provider deepseek --cache-model deepseek-flash

$PY -m verifacto.main greedy --detector replay --runs 5 --k 10 --pool 60 \
  --cache-provider deepseek --cache-model deepseek-flash
```

`--cache-provider` and `--cache-model` are optional, but recommended: they fail loudly if
an entry came from a different model instead of silently mixing verdicts.

Replay verifies the exact claim/language/source/policy cache key. If it raises
`DetectorCacheMiss`, the corpus or cache changed; rerun `precompute`.

### Current cache caveat

An offline audit against the fixed material-fact guard found five pre-fix fast-path claims
that would now be rejected (posts 16, 26, 37, 39, 61). All five have `UNCERTAIN` verdicts,
so no decided `FAKE`/`REAL` replay result is affected. A full rebuild would recompute them.

## 5. Track C — rebuild the real detector cache (optional, live)

Only do this when you deliberately want fresh live verdicts. It calls the real detector for
every post and takes roughly **one hour for 80 posts**, plus provider cost/quota.

`precompute --recompute-stale` rebuilds only entries whose policy version, provider, or
model differs, or whose cached run recorded a provider failure. It **does not** force
rebuilds of current entries. To rebuild the whole corpus, move the existing cache aside:

```bash
mv verifacto/data/detector_cache.json \
   "verifacto/data/detector_cache.manual-$(date +%Y%m%dT%H%M%S).json"

# Optional: also clear the semantic vector cache so no claim is reused.
# Without this, the fixed guard may still legitimately reuse equivalent claims.
mv pipeline/.qdrant_storage \
   "pipeline/.qdrant_storage.manual-$(date +%Y%m%dT%H%M%S)"

# Live run over the whole corpus
$PY -m verifacto.main precompute
```

Start with a smaller batch if the provider quota is uncertain:

```bash
$PY -m verifacto.main precompute --max-posts 5 --pacing 3
```

Important:

- `precompute` skips posts already cached and current. If the first five are current, a
  `--max-posts 5` run will skip all five and make no live calls.
- To force a live check without touching the cache, use a direct real-mode command:

```bash
$PY -m verifacto.main targeted --detector real --allow-live \
  --max-posts 5 --runs 2 --k 5
```

- Direct real-mode commands process at most `--max-posts` (default 10).
- Degraded “Pipeline error” verdicts are never cached. Three consecutive failures abort the
  run; fix the provider keys/quota and resume.
- `precompute` is resume-able and rewrites the cache after every successful post.

After rebuilding, rerun the Track B replay commands.

## 6. Optional analyses for Steps 5 and 6

```bash
# Cross-language bridge detection and Louvain sanity check (console only)
$PY -m verifacto.main bridges

# Rank flagged posts by confidence x expected reach
$PY -m verifacto.main priority --detector stub --runs 10

# Greedy throttling heuristic vs betweenness vs baseline
$PY -m verifacto.main greedy --detector stub --runs 5 --k 10 --pool 60

# Unified comparison at several budgets (includes greedy; slower)
$PY -m verifacto.main compare --detector stub --runs 30 --budgets 10 25
```

Artifacts:

- `bridges`: console report.
- `priority`: `priority.json` plus `priority_stub.json`.
- `greedy`: `greedy_comparison.png`, `greedy_metrics.json`, mode copy.
- `compare`: `strategy_comparison_k10.png`, `strategy_comparison_k25.png`,
  `compare_metrics.json`, mode copy.

## 7. Sensitivity sweeps

Sweeps use the stub detector only.

```bash
$PY -m verifacto.main sweep p --values 0.05 0.1 0.15 --runs 30
$PY -m verifacto.main sweep k --values 5 10 25 50 --runs 30
$PY -m verifacto.main sweep accuracy --values 0.7 0.8 0.9 --runs 30
```

Add `--with-greedy` only if you can wait; greedy selection is the slow part. Outputs are
`sweep_p.png/json`, `sweep_k.png/json`, and `sweep_accuracy.png/json`.

## 8. What a successful run looks like

- `graph` reports clear language clusters and a small cross-language edge fraction.
- `baseline` has the highest fake-news reach.
- `blanket` is clearly below the baseline but has a measurable real-news false-positive
  cost.
- `targeted` places the targeted strategies between blanket and baseline; betweenness is
  usually the best targeted heuristic.
- Replay mode reports the same qualitative ordering using real verdicts.
- `metrics.json` contains `meta.detector_mode`; the mode-specific copy
  (`metrics_stub.json`, `metrics_replay.json`, `metrics_real.json`) is the safer artifact
  to keep for comparisons.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: verifacto` | You are not in the repository root. `cd` there first. |
| `DetectorCacheMiss` | The cache is missing, or the post text/language/source changed. Rerun `precompute`. |
| Real mode aborts immediately | Add `--allow-live`. |
| Live results vary between runs | Expected: live mode is non-deterministic. Use stub or replay for comparisons. |
| `precompute` aborts after three failures | LLM/provider quota or keys. Fix `.env` and resume. |
| `precompute --recompute-stale` skips everything | Entries are already current. Move the cache aside for a full rebuild. |
| Replay provider/model mismatch | Use the same `--cache-provider/--cache-model`, or rebuild the cache. |
| Hindi/French text garbles in the Windows console | `export PYTHONIOENCODING=utf-8` (Git Bash) or set `$env:PYTHONIOENCODING="utf-8"` (PowerShell). |
| Charts or metrics missing | Check the command’s console output; the run may have aborted before writing. |

## 10. Minimal copy-paste core

The shortest complete offline run:

```bash
$PY -m verifacto.main graph
$PY -m verifacto.main posts
$PY -m verifacto.main baseline
$PY -m verifacto.main blanket
$PY -m verifacto.main targeted --detector stub --runs 30 --k 25
```

The shortest real-verdict run, assuming the existing cache is current:

```bash
$PY -m verifacto.main targeted --detector replay --runs 30 --k 25 \
  --cache-provider deepseek --cache-model deepseek-flash
```
