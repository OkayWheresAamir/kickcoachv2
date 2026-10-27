# KickCoach v2 — Start Here (for Claude Code)

You are helping Aamir rebuild **KickCoach** into a properly engineered, evaluated AI project. Read this file first, then the others in order.

## Read order
1. `00_START_HERE.md` — working rules (this file)
2. `01_PROBLEM_BRIEF.md` — what we're building and why
3. `02_MVP_SPEC.md` — scope, inputs and outputs, acceptance criteria
4. `03_ARCHITECTURE_AND_DECISIONS.md` — components, data flow, decisions and the alternatives rejected, risks
5. `04_EVALUATION_PLAN.md` — datasets, ground truth, metrics, baselines
6. `05_IMPLEMENTATION_ROADMAP.md` — vertical slices in build order, with time boxes

## Context
- **This is a fresh repo for KickCoach v2.** The old code is `kickcoach-2`, a continuation of the hackathon repo `kickcoach-synapse.ai`. Read it, run it if you can, and port only what is useful.
- **Old stack:** Python, FastAPI, TFLite MoveNet (17 keypoints), OpenCV, NumPy, Gemini REST, static HTML/JS frontend, CSV storage.
- **Known v1 problems:**
  - no tests and no requirements file;
  - the frame rate is assumed instead of read;
  - one saved row has an impossible knee velocity of 16,444°/s;
  - "hip rotation" is a feature, but it can't be observed from a side view;
  - comparison against pro "archetypes" built from YouTube clips that were never validated.
- **The old repo has upgrade-planning documents from July 2026.** Read them. Where they conflict with this handoff, **this handoff wins on scope**. List the conflicts for Aamir; don't silently merge them.
- **Time box:** a resume-ready version in about one day of work. Phase 1 of the roadmap is the resume version. Everything after that is optional improvement.

## Non-negotiable rules
1. **Never create ground-truth labels yourself.** You may build the labelling tool and pre-fill keypoints from a pose model. Aamir corrects and confirms every label. Model-made labels used as ground truth make the whole evaluation circular.
2. **Never invent numbers.** This covers biomechanical norms, fault thresholds, sanity bounds, and evaluation results.
   - Every threshold or norm must cite a source (paper, table, page), or be explicitly marked `heuristic` both in code and in the user interface.
   - Results in the README come only from scripts that actually ran.
   - Targets are never written as if they were achieved.
3. **The LLM never measures.** Angles, timings, positions, and the comparisons come from the pose model plus deterministic code. The LLM writes feedback, assists with extracting norms offline (a human verifies), and is evaluated as a *baseline*.
4. **Never commit** video clips, extracted frames, API keys, or `.env`. `data/raw/` and `data/frames/` are gitignored. Commit the manifest (URLs plus timestamps) and the label files (coordinates only).
5. **The app must run without an API key.** LLM features fall back to templates.
6. **Check current package APIs before using them.** This applies to rtmlib/RTMPose, ultralytics, mediapipe, google-genai, scipy, and others: check the official docs or installed source, because APIs change.

## How to work
Every important component must be explainable in plain terms. For each slice in the roadmap:

1. State the behaviour you're implementing and its acceptance criteria.
2. Explain the 1–3 concepts the slice needs (for example: why filter before differentiating, or what PCK measures). Keep it short and concrete.
3. Propose the smallest sensible change. Flag risks and assumptions.
4. Implement it with tests.
5. Run the tests or scripts and **show the real output**, including failures.
6. Diagnose failures from evidence: logs, intermediate arrays, plots. Don't make speculative edits.
7. Add an entry to `docs/DECISIONS.md` (decision, alternatives, evidence, consequence), plus a 2–3 line plain-language summary.

**Ask Aamir before you:**
- pick a pose model when the evidence is close;
- drop or add a feature;
- change scope;
- spend money on API calls beyond small tests;
- download anything other than the clips in the manifest.

## Proposed repo layout
```
kickcoach-v2/
  docs/            # these handoff files + DECISIONS.md
  kickcoach/
    ingest/        # video loading, fps/metadata, frame extraction
    pose/          # model adapters -> canonical skeleton
    signal/        # gap fill, leg-swap fix, filtering, derivatives
    events/        # kicking leg, contact frame (ball-based), phases
    features/      # feature definitions + sanity bounds
    quality/       # viewpoint / visibility gate
    compare/       # norms table, fault rules, self-history
    feedback/      # LLM feedback (structured) + validator + templates
    retrieval/     # Phase 2: coaching corpus RAG
    storage/       # SQLite
  api/             # FastAPI app
  tools/           # labelling tool, clip fetcher, frame extractor
  eval/            # one script per eval (E1..E8), writes to results/
  knowledge/       # norms.csv, faults.yaml, sources.md, corpus/ (Phase 2)
  data/            # manifest.csv, labels/ (committed); raw/, frames/ (ignored)
  results/         # eval outputs (CSV/JSON + small plots)
  tests/
```
