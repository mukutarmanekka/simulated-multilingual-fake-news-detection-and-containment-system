---
name: verifacto-containment
description: How to build and evaluate the network-containment layer of the VeriFacto project, a simulated social network where a fake news detector triggers actions that reduce misinformation spread. Use this skill whenever the user works on VeriFacto, a simulated social network, misinformation spread or cascade simulation, Independent Cascade or SIR models, containment or blocking of fake news, influence maximization, centrality-based intervention (betweenness, PageRank), greedy or CELF selection, or asks what to do with a fake news detector's output, even if they do not say "containment" explicitly.
---

# VeriFacto Containment

Turns a working fake news detector into a closed loop on a simulated social network:

    detect -> model spread -> decide intervention under a budget -> act -> measure -> feed back

The detector is a black box that returns `(label, confidence)`. This skill covers everything after it.

## Workflow (follow in order)

Never start a step before the previous one works. Use one fixed spread probability and the Independent Cascade model until Step 5 is done.

### Step 1: Build the graph

- Use NetworkX, a directed graph, about 500 users.
- Give each user a `language` attribute.
- Use a stochastic block model with one block per language: dense inside blocks, sparse between blocks. Add a few hub users (high out-degree) and a few bilingual bridge users that connect blocks.
- Give every edge one fixed spread probability `p` (start at 0.1).
- Check: print block sizes and the fraction of cross-language edges.

### Step 2: Build the post list

- 50-100 posts, each with `text`, `true_label` (hidden), `author`, `language`.
- The author must belong to the post's language block.
- Keep the real/fake ratio explicit so results are interpretable.

### Step 3: Simulate spread with no protection

Independent Cascade, one post at a time:

1. Active set starts as the author. Newly active users get exactly one chance to infect each out-neighbor.
2. Each attempt succeeds with the edge probability.
3. Newly infected users become the next time step's frontier.
4. Stop when the frontier is empty.
5. Record the count of newly reached users per time step.

Use a seeded random generator. This is the baseline everything else is compared to.

### Step 4: Plug the detector in

- When a post appears, call the detector.
- If flagged fake, apply the simple action: multiply the post's edge probabilities by a reduction factor (for example 0.5), or set it to 0 for a full stop.
- Rerun Step 3 on the same graph, same posts, same seeds.
- Plot reach-over-time with and without detection.

### Step 5: Targeted containment

Instead of reducing spread everywhere, throttle only k users.

- Heuristic selections: top-k by degree, betweenness, PageRank, plus a random-k control.
- Throttling a user means reducing the probability on their outgoing edges.
- Compare all strategies on the same graph with the same budget k.

### Step 6: Optional upgrades

- **Greedy selection:** an empirical heuristic, not a provably near-optimal method. The expected spread *reduction* as a function of the set of throttled users is NOT generally monotone submodular — blocking/throttling objectives are known to be non-submodular — so greedy carries NO (1 - 1/e) guarantee here. Do not use CELF's lazy bound either: it requires monotone submodularity to be valid. Use full reevaluation each round instead. Estimate each candidate's score by averaging many Monte Carlo cascades, on draws disjoint from the evaluation seeds so the selection does not overfit the runs it is scored on. Benchmark against the centrality heuristics rather than assuming greedy wins, and define the objective over flagged posts only (an operator does not know the hidden labels).
- **Cross-language bridges:** run community detection (for example Louvain). Users with edges into more than one language community are bridge nodes, and are natural containment points. When a claim is flagged in one language, check its translation in the other languages before it jumps over.
- **Counter-campaign:** pick k seed users to spread the correction, framed as influence maximization of the truth cascade. This objective (expected spread from a fixed seed set) IS monotone submodular, so greedy here does carry the (1 - 1/e) guarantee and CELF applies.
- **Dashboard:** live graph colored by exposed / protected state.

## Prioritizing flagged posts

When the intervention budget is limited, rank flagged posts by expected harm:

    priority = detector_confidence x expected_reach

where `expected_reach` is the Monte Carlo mean cascade size from the post's author.

## Action ladder (extension of Step 4)

| Situation | Action |
|-----------|--------|
| Low confidence | Attach a "disputed" label |
| Medium confidence | Add share friction, downrank, limit forwarding |
| High confidence and high reach | Temporary hold, push a correction to exposed users |
| Repeat spreader | Lower that user's outgoing edge weights (reputation score) |

Avoid hard deletion at low confidence.

## Evaluation protocol

For every strategy, on identical graphs, posts, and seeds:

- Final fraction of users reached by fake posts
- Peak spread and the time step it occurs
- Time to containment
- Number of interventions used
- Real posts wrongly flagged or suppressed

The headline result is a plot of spread over time for: no intervention, random, degree, betweenness, and greedy. Average over many seeds and report the spread (mean and standard deviation or a confidence band), not a single run.

## Common pitfalls

- Letting the detector see the true label. The label is only for metrics.
- Tuning `p` or switching models early. Fix them until Step 5 is done.
- Comparing strategies on different random graphs or seeds.
- Reporting only fake news stopped, not real news wrongly suppressed.
- Reporting a single simulation run. Cascades are random, so average many runs.
- Claiming a (1 - 1/e) guarantee for greedy throttling. That bound holds for seed-set influence maximization (the counter-campaign), not for blocking/throttling.
