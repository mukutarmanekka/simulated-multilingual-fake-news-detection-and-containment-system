# VeriFacto: Project Guide for Claude

## What this project is

B.Tech capstone project at NIT Delhi.

The faculty idea: "Everyone detects fake news, but nobody acts on it." So this project has two halves:

1. **Detection (DONE):** a real-time, multilingual fake news detector.
2. **Action (TO BUILD):** a simulated mini social network where every detection triggers a decision, and the decision measurably reduces the spread of fake news.

The whole system is a closed loop:

    detect -> model spread -> decide intervention under a budget -> act -> measure -> feed back

The detector is only the sensor. The graded contribution is the network layer around it.

## Current status

- Done: multilingual real-time detector (treat it as a black box, see "Detector interface").
- Not started: network, spread simulation, interventions, containment optimization, evaluation, dashboard.

## Build order (strictly linear)

Do the steps in order. Do not start a step until the previous one meets its "done when" condition. Save a working version after each step.

| Step | Task | Done when |
|------|------|-----------|
| 1 | Build a ~500-user directed graph with NetworkX. Each user has a language tag. Most edges stay inside a language, a few bilingual users bridge languages. | Printing the graph shows clear language clusters. |
| 2 | Build a list of 50-100 posts (real and fake) in the supported languages, each with text, hidden true label, and an author in the matching language. | Every post has text, label, author. |
| 3 | Spread posts with no protection. A post starts at its author and each informed user passes it to each neighbor with probability `p` per time step. | A "users reached over time" plot exists for fake posts (the baseline). |
| 4 | Plug the detector in. If a post is flagged fake, reduce its spread probability (halve it or stop it). Rerun Step 3. | Two curves (with and without detection) are plotted, and the protected one is lower. |
| 5 | Replace the blanket action with targeted containment: pick the top k users to throttle by betweenness or PageRank, compare against random. | One chart compares random, degree, and betweenness selection. |
| 6 | Polish, only if time remains: greedy selection (benchmarked empirical heuristic, no submodularity guarantee), cross-language bridge detection, a dashboard. | - |

Steps 1 to 4 already make a complete, demonstrable project. Step 5 and 6 raise its quality.

## Rules that keep the project simple

- Use ONE fixed spread probability (start with `p = 0.1`) and ONE spread model (Independent Cascade) until Step 5 is finished. Do not tune early.
- The detector never sees the ground-truth label. The simulator uses the hidden label only to compute metrics afterwards.
- Every random operation must use a fixed seed so experiments are reproducible.
- Compare every strategy against the no-intervention baseline on the same graph and the same posts.
- Wrongly suppressing real news is a cost. Always measure it, do not only measure how much fake news was stopped.

## Suggested layout

    verifacto/
      detector/            existing detector, do not modify without asking
      network/
        graph_builder.py   Step 1
      data/
        posts.py           Step 2
      simulation/
        cascade.py         Step 3
        intervention.py    Step 4
      containment/
        heuristics.py      Step 5 (degree, betweenness, PageRank, random)
        greedy.py          Step 6 (greedy heuristic, full reevaluation)
      evaluation/
        metrics.py
        plots.py
      main.py

## Detector interface

Treat the detector as a function. Assumed signature (confirm with the user and adjust if the real one differs):

    detect(text) -> (label, confidence)

`label` is "fake" or "real", `confidence` is a float in [0, 1]. Wrap it in one adapter function so the rest of the code never depends on the detector's internals.

## Metrics to report

- Final fraction of users who received each fake post
- Peak spread and time step at which it occurs
- Time to containment
- Number of interventions used (the budget)
- Real posts wrongly flagged or suppressed (false-positive cost)

## Tech stack

- Python 3
- NetworkX (graphs), NumPy, Matplotlib
- Install: `pip install networkx numpy matplotlib`

## How to write code here

- Preserve the original variable names already in the code.
- Improve the existing approach instead of replacing it. Avoid unnecessary rewrites.
- Write readable, well-organized code. Use plain `if/else` blocks. Do not use ternary operators or dense one-liners.
- Add short comments only where the logic is not obvious.

## How to talk to the user

- For a clear task, be direct and skip preamble.
- When explaining code, tie the explanation to the exact lines in front of you. Go through the mechanism step by step and do not skip intermediate reasoning. Say what changes internally, not just that "it works".
- When asked to verify something, give a short confirmation first, then details only if needed.
- For strategy discussions, use plain prose. For pipelines, use a structured breakdown.
