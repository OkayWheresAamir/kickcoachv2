# CLAUDE.md

KickCoach v2 measures an instep kick from side-on video using a pose model plus deterministic code. It compares the measurements with cited norms and the player's own history, and writes feedback with an LLM under a validator. Every measurement is evaluated against human labels, with a frontier video model as a baseline. It's a rebuild of a hackathon project.

Personal working notes, when present, are in a gitignored file: @CLAUDE.local.md

## Docs
- `docs/PLAN.md`: the phases. `docs/DECISIONS.md`: every decision (DEC-NNN). `docs/JOURNAL.md`: the build log.
- `docs/handoff/` is reference material. Where it conflicts with `docs/PLAN.md` or `docs/DECISIONS.md`, those two win. The audit of v1 is in `docs/V1_AUDIT.md`.
- **The repo is public.** Tracked files record project facts only: no personal notes, working-style remarks, machine details or third parties' contact details.

## Non-negotiable rules (from docs/handoff/00_START_HERE.md)
1. **Never create ground truth.** That covers keypoint labels, contact and backswing frames, the kicking leg and `view_ok_human` (DEC-007). Pre-filling and QA flags are fine; setting or editing a confirmed value is not.
2. **Never invent numbers.**
   - Every threshold, norm or bound cites a source or is marked `heuristic` in both the code and the UI.
   - README numbers come only from scripts that ran.
   - Targets are never written as results.
3. **The LLM never measures.** It writes feedback, helps extract norms offline (a human verifies them), and is evaluated as a baseline.
4. **Never commit** video, frames, API keys, `.env`, model weights, or personal documents.
5. **The app runs without an API key.** LLM features fall back to templates.
6. **Check current package APIs before use.** Read the docs or the installed source; don't rely on memory.
7. **Gemini is free tier only (DEC-003).** Use the rate-limited, cached client.
8. **Commits are made by hand** (DEC-005): never run `git commit` or `git push`.

## Commands
```bash
uv sync                    # install (creates .venv)
uv run pytest              # tests
uv run ruff check . && uv run ruff format --check .
uv run python -m eval.run_all   # every eval -> results/summary.md + the README's results table (no API calls)
uv run python tools/audit_v1/reconstruct_history_scramble.py   # audit evidence
uv run python tools/rate_feedback.py        # the E7 human-rating page: http://127.0.0.1:8766/
uv run python -m eval.e7_feedback judge     # the LLM judge (resumable; free-tier quotas, DEC-040)
uv run python -m eval.e9_chat run       # E9: the coach chat's scripted questions (cached; resumable)
uv run python tools/coach_chat.py --player me   # the coach chat in the terminal (local sessions)
uv run uvicorn api.app:app --port 8000   # the app: http://127.0.0.1:8000/ (runs without an API key)
uv run --group deploy modal deploy deploy/modal_app.py   # redeploy the live demo (DEC-035)
uv run --group deploy python deploy/smoke_test.py <url> data/raw/clips/c07.mp4   # real upload test
swiftc -O tools/snap_page.swift -o /tmp/snap_page   # page screenshots (macOS WebKit)
```

## Layout
Packages are created as their phase arrives.
```
kickcoach/   ingest/ pose/ llm/ signal/ events/ features/ quality/ compare/ feedback/ retrieval/ chat/ storage/
api/         FastAPI app            tools/     labelling tool, clip fetcher, audit scripts
deploy/      Modal app (live demo) and an upload smoke test
eval/        one script per eval    knowledge/ norms.csv, faults.yaml, sources.md
data/        manifest.csv, labels/ (committed); raw/, frames/, cache/ (ignored)
results/     eval outputs, incl. cached LLM responses (committed)     tests/
```

## Conventions
- **Pose arrays:** `[T, K, 3]` of `(x, y, conf)` in source-frame pixels, with y down.
- **Angles:** in degrees. Trunk lean is positive = forward in the kicking direction (DEC-006). Degenerate geometry gives NaN.
- **Values that fail a sanity bound** become `not_measured` with a reason. They are never clamped.
- **Comments** explain why, not what. Keep modules small and pure where possible, so they're testable.
