# KickCoach v2: phase plan (one phase per chat)

This plan turns the handoff roadmap (`docs/handoff/05_IMPLEMENTATION_ROADMAP.md`, slices S0–S16) into chat-sized phases. Where they differ, this file and `docs/DECISIONS.md` win. The handoff docs remain the reference for *what* each slice must do.

- **Phases 0–6** = the handoff's "Phase 1", the resume version.
- **Phases 7–9** = the extras (DEC-008).
- **Phase 10** = the write-up; **Phase 11** = the system evaluation, which takes in SportsPose's E10 (DEC-044).

Each phase ends with passing tests and lint, and updated DECISIONS, JOURNAL and PLAN.

## Phase 0: audit and setup ✅ (2026-10-27)
- **Slice:** S0.
- **Done:**
  - `docs/V1_AUDIT.md`;
  - repo skeleton (uv, pytest, ruff), `CLAUDE.md`, this plan, `STATUS.md`, `DECISIONS.md`;
  - first tested modules: `kickcoach/features/geometry.py` and `kickcoach/pose/letterbox.py`.

## Phase 1: clips, ingest, pose adapters ✅ (2026-10-28)
- **Slices:** S1 and S2.
- **Done:**
  - **Manifest:** 26 clips (23 side-on + 3 rear-view gate negatives), approved by Aamir from the contact sheet; 5 held out. Every candidate and the reason for keeping or rejecting it is in `data/candidates.csv` (DEC-009).
  - **Fetch and ingest:**
    - `tools/fetch_clips.py` downloads the sources. Trimming moved to `tools/ingest_clips.py`, because without an ffmpeg CLI yt-dlp can't cut sections.
    - `kickcoach/ingest/` writes frames, a frame-aligned trimmed MP4, metadata and `fps_trusted` (DEC-010).
  - **Pose:** MoveNet, RTMPose-m Halpe26 and MediaPipe heavy, all into a 23-point skeleton, cached for all clips (DEC-011).
  - **Audit follow-ups:** V1_AUDIT 2.2 confirmed on real clips; new findings 2.8 (the kick gate never fires) and 2.9 (BGR input); 2.5 is superseded.
  - **Gemini:** client, prompt and a 3-clip gut-check (DEC-012); see `results/gemini_gutcheck/summary.md`.
- **Deviations:**
  - 3 approved clips were dropped after download (drawn overlays at contact);
  - `slow_motion` was added as Claude's assessment with its basis in `notes`;
  - the manifest has two extra columns, `split` and `purpose`.

## Phase 2: labelling tool and guide ✅ (2026-10-28)
- **Slice:** S3.
- **Write `docs/LABELLING_GUIDE.md`:**
  - one-paragraph definitions of each landmark (hip, knee and ankle joint centres; shoulders; heel; toe; ball centre), taken from cited sources: the COCO keypoint definitions plus a 2D sagittal-plane biomechanics reference. Find and cite them; never write them from memory;
  - the contact frame: the first frame in which the foot touches the ball;
  - the max-backswing frame: the frame of deepest knee flexion before contact;
  - how to mark a point that is occluded instead of guessing;
  - example images.
- **Build `tools/label_tool.py`.** Choose a local browser UI or a matplotlib one, whichever is more reliable on macOS.
  - **Pre-fill:** keypoints from RTMPose (the only model with all 6 foot points; E1 hasn't run yet, so "best" isn't known). Aamir drags any wrong point and confirms each frame.
  - **Clips to label:** the 23 `purpose=eval` clips get keypoints. The 3 `gate_only` clips only need `view_ok_human` (and the contact frame, if visible).
  - **Clip-level fields:** frame scrubbing to mark contact and max backswing, plus kicking leg and view OK.
  - **Logging:** per-frame time.
  - **Repeatability mode:** hides previous labels and samples 10 frames.
  - **Saving:** `data/labels/<clip_id>.json`, coordinates only.
- **QA flags shown to Aamir** (Claude never edits a label):
  - thigh or shank length inconsistent across frames of a clip;
  - a large disagreement with the model;
  - a missing required point.
- **Accept when:** Aamir has labelled 2 trial clips in-session and the tool works for him.
- **Done:**
  - `docs/LABELLING_GUIDE.md`, with cited definitions (ISB joint centres, a 2D side-view surface-landmark method, OpenPose / COCO-WholeBody foot points, Kellis & Katis events, COCO visibility flags) and two schematic figures (`docs/img/`);
  - `tools/label_tool.py` + `tools/label_ui/` + `kickcoach/labels/` (schema, QA, timing); 28 tests (DEC-013);
  - trial: c10 and c01 fully labelled by Aamir; budget set from the measured times (DEC-014).
- **Deviations:**
  - 2 extra clip fields: `kick_direction` (needed for the trunk-lean sign) and `contact_between_frames`;
  - small toes not labelled;
  - contact − 1 made optional (DEC-014);
  - from Aamir's feedback: whole-clip browsing in the keypoint screen, and a bigger notes box.

## Phase 3: E1 and the measurement core ✅ (2026-10-30)
- **Slices:** S4 and S5, in one chat (no 3a/3b split).
- **Done:**
  - **E1** (`eval/e1_pose.py` → `results/e1*.csv`, `results/e1_summary.md`): PCK@0.1, normalised error and missing rate, by joint group, frame role, visibility and split. It adds a clip bootstrap, a moved-points-only anchoring check, raw per-frame angle error and leg-swap counts. **RTMPose-m chosen** under a rule fixed in advance (DEC-016).
  - **Signal** (`kickcoach/signal/`): confidence mask → Viterbi leg-swap fix → segment-length glitch mask → gap fill → Butterworth `sosfiltfilt` (18 Hz cited; 0.15 cycles/frame when timing is unknown) with a break at contact → central differences (DEC-019). Cutoff tuning on the tuning split: `eval/tune_cutoff.py`.
  - **Events** (`kickcoach/events/`): YOLOX-m ball candidates (`tools/run_ball.py`). Contact from the ball first (motion → bracket → foot speed), then the kicking leg by displacement into the ball, kick direction, and the max-backswing valley (DEC-018, DEC-020).
  - **Features** (`kickcoach/features/kick.py`, `bounds.py`): 6 features, cited or heuristic bounds, velocities and ms timing only at a trusted ≥ 120 fps. **Gate** (`kickcoach/quality/gate.py`): view ratio, keypoint score, kick evidence (DEC-021).
  - **Pipeline** (`kickcoach/pipeline.py`, `tools/run_pipeline.py` → `results/pipeline/`, `results/pipeline_summary.csv`): runs on all 26 clips; 0 values beyond a bound.
  - **Tests:** a synthetic stick-figure kick (`tests/synthetic_kick.py`; known angles, contact, a 1000°/s velocity kink at impact) at 30–240 fps, with noise, in both directions; plus leg swaps, bounds, refusal and E1 metrics. 139 tests pass (64 new).
- **Deviations:**
  - `ankle_angle_contact` dropped (Aamir, DEC-017);
  - extra signal steps: a segment-length glitch mask, a contact break, and the swap fix before the gap fill;
  - the kicking leg comes from the ball contact, not the whole-clip peak foot speed (D8's rule failed on 7 of 19 clips);
  - a missing ball and untrusted timing don't refuse a clip: they withhold the affected features.

## Phase 4: evaluations E2, E3, E8 and the frontier baseline E4 ✅ (2026-10-31)
- **Slices:** S6 and S7.
- **Done:**
  - **v1 baseline:** `eval/v1_replica.py` + `tools/run_v1_replica.py` replay v1 as shipped (BGR, letterbox bug, EMA, fallback "kick" frame); the plan's "v1-style" raw MoveNet is kept as an ablation (DEC-024).
  - **E2** (`eval/e2_angles.py` → `results/e2*.csv`, `results/e2_summary.md`): end to end (v1 vs v2), at the labelled frames (model × cleaning), the event/measurement decomposition with the winner's curse, anchoring, cutoff sensitivity (DEC-025).
  - **E3** (`eval/e3_contact.py` → `results/e3.csv`, `results/e3_summary.md`): ball vs foot speed (blind and given the leg) vs v1; the bracket correction adopted by a pre-stated rule (DEC-022).
  - **E8** (`eval/e8_reliability.py` → `results/e8_*.csv`, `results/e8_summary.md`): bound violations v1 vs v2, gate vs `view_ok_human`, velocities withheld.
  - **E4** (`eval/e4_gemini.py` → `results/e4_raw/`, `results/e4.csv`, `results/e4_summary.md`): `gemini-3.8-flash`, prompt e4-v2, HIGH media resolution and thinking, 6 repeat calls (DEC-023). **Partial at first:** the free tier allows 20 requests a day for this model, and 10 of the 32 were answered on day 1. `collect` resumes in priority order. **Complete on 2026-11-09:** all 32 requests answered (26 of 26 clips).
  - **Failure analysis:** `eval/failures.py` renders the worst cases to `data/failures/` (local); diagnoses in `results/failures.md`.
- **Deviations:**
  - E2's baseline is v1 replayed, not only "MoveNet unfiltered" (DEC-024);
  - E4 also asks for max knee flexion (prompt e4-v2), and runs on all 26 clips (the 7 without labels show refusal behaviour only);
  - pipeline change: the bracket contact shift (DEC-022), and a `target_cpf` override in `kickcoach.pipeline.analyse` for the sensitivity sweep.

## Phase 5: knowledge, LLM feedback, storage, API ✅ (2026-11-02)
- **Slices:** S8 and S9. Both are required (DEC-002).
- **Done:**
  - **E4 finished for every scored clip** (19 of 19 asked, 18 answered; DEC-023). 5 unlabelled clips (refusal behaviour only) remain for any later day.
  - **Norms** (DEC-026): `tools/extract_norms.py` extracted 32 rows from 4 open-access papers plus 2 figures-only controls with `gemini-3.6-flash`. Aamir verified them in-session (`knowledge/norms.csv`). **E5:** 0 hallucinated rows, 512/512 fields, recall 94% (`eval/e5_norms.py` → `results/e5_summary.md`).
  - **Comparison engine and faults** (DEC-027, DEC-028): `kickcoach/compare/`, `knowledge/faults.yaml` with 4 rules, each cited or heuristic, one threshold derived from a verified norm.
  - **Feedback** (DEC-029): `kickcoach/feedback/`, structured output → validator → one retry → template. 15/15 first-try passes on the corpus (`tools/run_assess.py --llm` → `results/assess_summary.csv`, `results/feedback_llm/`).
  - **Storage and API** (DEC-030): `kickcoach/storage/db.py` (SQLite, self-history), `api/app.py` (FastAPI), the adapted v1 page `api/static/index.html`. A real upload works end to end, and the app runs with no API key (tests).
- **Deviations:**
  - 2 control papers on top of the 2–4 sources, to measure hallucination (DEC-026);
  - the extraction model is `gemini-3.6-flash`, not the lite dev model (quota and quality; DEC-026);
  - `tools/run_assess.py` adds a corpus-level run of comparisons, faults and feedback (not in the handoff);
  - the page's rendering wasn't visually checked (headless Brave failed in this session).

## Phase 6: results, README, resume ⇒ resume-ready ✅ (2026-11-02; the floor waits on the blind relabel)
- **Slice:** S10.
- **Results:** `eval/run_all.py` regenerates `results/summary.md` from the committed labels and cached outputs (E1–E5, E8, plus `tools/run_assess.py`'s fault and feedback summary). No API calls: every LLM result comes from committed caches.
- **README:** problem, architecture, capture protocol, results with n, the labelling repeatability floor, decisions, limitations, and how to run.
- **Summary:** draft the project summary from the actual numbers (template in eval plan §"Resume claims").
- **Accept when:** a fresh clone plus the documented steps reproduces the summary.
- **Done:**
  - **`eval/run_all.py`** (DEC-032): 10 steps in 34 s with the API key blanked → `results/summary.md` and the README's generated results table. A copy of exactly the tracked files (no `.env`, video or frames) reproduced every file in `results/` byte for byte (accepted).
  - **Reproducible inputs** (DEC-031): pose keypoints, ball detections and per-clip `meta.json` are committed; E4's cache is found by tag when the video is absent.
  - **`eval/repeatability.py`** (DEC-033): the floor from a blind relabel, all points and moved-only. **Not measured yet:** the relabel is still to do; the README says so.
  - **README** rewritten; the project summary drafted from the actual numbers (kept outside the repo).
- **Deviations:**
  - derived caches committed (DEC-031), which the plan didn't foresee;
  - the README's results table is generated, not hand-written;
  - E4's last 5 (unlabelled) clips weren't asked: the free-tier quota was still used up at 14:26 IST (they were asked on 2026-11-09);
  - the page still wasn't looked at (headless Brave fails here); its JS fields were cross-checked against real API responses instead.

## Phase 7: live demo ✅ (2026-11-02)
- **Slice:** S13.
- **Deploy** the backend and frontend, with an upload size limit and videos deleted after processing.
- **Check the free-tier data terms first.** Free-tier Gemini inputs may be used by Google, so for strangers' uploads either default to template feedback or show a clear notice. Confirm the current terms before deciding.
- **Done:**
  - **Live** at https://okaywheresaamir--kickcoach-web.us-east.modal.run (`deploy/modal_app.py`, Modal free credits; DEC-035). Smoke-tested with real uploads: accepted, portrait-phone, refused and too-long clips.
  - **Terms:** the free tier may not serve users in the EEA, Switzerland or the UK at all, so the public mode (DEC-034) forces template feedback and stores nothing server-side; history lives in the visitor's browser.
  - **Jobs:** uploads are queued and polled (Modal caps a request at 150 s), with real per-frame progress. The 100 MB / 600-frame limits stay, and uploads are deleted after processing.
  - **Page rebuilt** (DEC-036): starts empty, an opt-in example (c07 from committed caches), a stick figure from the measured keypoints, light/dark/phone layouts, checked by screenshot (a Swift WebKit snapshot tool).
  - **Ingest fix:** phone display rotation is applied, so portrait uploads aren't measured sideways.
- **Deviations:**
  - the host is Modal, not a "free hosting account" in the HF sense: HF now charges for Docker Spaces;
  - the page was rebuilt rather than adapted (Aamir's review), and gained a stick figure and an example (approved);
  - history moved from a typed player name in SQLite to the browser (privacy on a public server); local mode still stores sessions;
  - the rotation fix is an ingest change the plan didn't foresee (no committed result changed).

## Phase 8: RAG coaching corpus and LLM-as-judge ✅ (2026-11-03; E7 closed 2026-11-07 without a human rater, DEC-041)
- **Slice:** S11, extended.
- **Retrieval (E6):** 30–60 tagged chunks, each with a source URL. Compare a metadata filter alone with filter + embeddings on Recall@3 and MRR.
- **Feedback faithfulness (E7):**
  - Aamir rates about 20 outputs;
  - an LLM judge rates the same outputs, and its agreement with Aamir is reported (e.g. weighted kappa / Spearman). The judge is used beyond those 20 only if the agreement is acceptable;
  - also report the validator's first-try pass rate.
- **Add to the resume only after E6 and E7 have run.**
- **Done:**
  - **Corpus** (DEC-037): 55 verbatim chunks from 10 approved sources, tagged by fault, kick type and level, each checked verbatim by a test. No sourced drill exists for 3 of the 4 faults.
  - **Retrieval** (DEC-038): a fault filter, then `gemini-embedding-2` cosine similarity. The vectors are committed, so runtime needs no key. **E6:** filter + embeddings Recall@3 0.56 / MRR 0.75; filter alone 0.46 / 0.61; embeddings alone 0.29 / 0.50 (9 queries; relevance labelled by Claude before any embedding).
  - **Feedback v3** (DEC-039): passages go to the LLM and become citable. v2 is byte-identical, and templates never retrieve. The validator rejected its first real answer (an invented citation). The page links cited passages. Redeployed.
  - **E7 protocol and tools** (DEC-040): 54 items in 3 variants, a frozen 20-item blind sample, the rating page, the judge (`gemma-4-31b-it`; 30 of 54 judged), scoring and a pre-set κ ≥ 0.6 bar. Validator first-try pass: v3 14 of 15 clips, v2 15 of 15.
- **E7, closed 7 Nov** (DEC-041): no human rating. LLM raters only: Claude (rules, not blind) and a supplied AI persona panel (blind). On 8 matched pairs, passages raised correctness in 7/8 and 6/8 and cost specificity. The judge (`gemini-3.6-flash`) gave correctness 5/5 to all 20, so it isn't usable for faithfulness. Not claimable: human validation of the judge.
- **Deviations:**
  - 3 approved sources dropped (2 bot checks, 1 basketball chart);
  - the judge moved from `gemini-3.6-flash` (20 requests a day) to Gemma 4 before any agreement existed;
  - the shank-before-thigh rule never fires, so E7 has no item for it;
  - an "embeddings only" row was added to E6 (cheap, and it informs Phase 9's free-text search).

## Phase 9: tool-calling coach chat ✅ (2026-11-08)
- **First:** the small validator and prompt fixes E7 found (STATUS, Next step 2).
- **What:** an open-ended Q&A ("why did my knee angle drop over 3 sessions?"). Gemini function calling over the tools `get_history`, `get_session_features`, `compare_to_norms` and `search_drills`. The deterministic validator is reused on its answers.
- **Separate from the pipeline:** the pipeline stays linear (D15 still holds there). The chat exists because open-ended questions need different data each time; see DEC-008.
- **Eval:** a scripted question set scoring tool-selection accuracy, numeric faithfulness, and refusal to discuss features marked `not_measured`.
- **Done:**
  - **E7's fixes** (DEC-042): two validator checks (a direction the measurement contradicts; a knee angle called bend) that reject exactly the 4 rated slips plus 1 unsampled. Prompt v4 puts the value in "what" (18 of 18 items, vs 3 of 18). E7's rated items are frozen.
  - **The chat** (DEC-043): four read-only tools that compute every number, manual function calling on `gemini-3.5-flash-lite`, and an answer checker (numbers, units and IDs grounded in tool results). Local only: `/chat`, a page panel and `tools/coach_chat.py`.
  - **E9:** 16 questions (1 added post hoc) on fixture players: right tools with the expected arguments 16 of 16, first-draft numbers grounded 55 of 55, checker first try 15 of 16, refusals 2 of 2. Three runs; the first two exposed missing units, an invented reason and an overclaimed trend, each fixed in the tools' output.
- **Deviations:**
  - the chat's eval is E9, so SportsPose's 3D check became E10;
  - a page panel was added (local mode only);
  - judge-v2 (optional) was not done;
  - the redeploy was blocked by the permission system, so it is run by hand.

## Phase 10: the write-up
- **Why now:** the system is built (Phases 0–9); the project summary is still the Phase 6/7 draft.
- **What:**
  - rerun `eval.run_all` and redraft the project summary from today's `results/summary.md`, now covering the live demo, retrieval and the chat (E6, E9). Every number gets its source file and n;
  - extend the "wording to avoid" list to Phases 8–9 (e.g. no "validated against human ratings"; E9 is 16 scripted questions on fixture players, not users);
  - make the README's opening match the summary.
- **Note:** the summary uses today's results. If Phase 11 produces a number on unseen phone video, it is updated then.

## Phase 11: system evaluation (the frozen system on unseen data)
- **Why:** E1–E9 each test one part, on 26 YouTube clips that were all looked at while building. The 5 held-out clips are person-disjoint but few, and mostly lab or slow motion. No real phone video has been through the app, the chat was tested on fixture players, and nobody has rated the feedback. This stage tests the finished product the way it will be used.
- **How:**
  1. **Freeze:** record the commit; no code or threshold changes during the stage. A failure is logged, fixed afterwards and retested on other clips.
  2. **Protocol first:** metrics, pass bars (each cited or marked heuristic) and failure categories are written to a file and hashed before any test clip is run, as E9's questions were.
  3. **Fresh test set:** new phone kicks filmed for this stage (several phones and frame rates; a few deliberately off side-on to test the gate), plus optional new YouTube clips (an ask-before download). Aamir labels them before the pipeline runs on them.
  4. **End to end,** through the live app's upload path and locally: the gate (false refusals, false accepts), the contact-frame error, and angle MAE with bootstrap CIs, side by side with the in-sample and held-out numbers. Also a breakdown by condition, latency, crashes and Modal cost.
  5. **Repeatability:** one player, about 5 kicks in a row. The per-feature spread against E2's error gives the smallest change the history and chat can call real. No labels needed.
  6. **Small user test** (handoff S16): 3–5 players upload their own kicks, read the feedback, ask the chat 3 questions and fill in a short form (clear? would you follow it? anything wrong?). E7's human rating too, if a rater is available. Consent first; nothing personal is committed.
  7. **E10, if SportsPose access is granted** (renumbered from E9 in Phase 9; DEC-043): 2D side-view angles from the pipeline vs 3D ground-truth joint angles, which quantifies the error inherent in 2D measurement by camera view. Caveat: SportsPose's ground truth is markerless (HRNet triangulated across 7 views), validated against Qualisys at a mean error of 34.5 mm. That is not marker truth, and it shares some bias with COCO-trained models; the README says so.
  8. **Report:** `results/system_eval.md`, a README section, and failure panels (`eval/failures.py`).
- **Smallest useful version:** Aamir's own 10 kicks plus 5 repeats, labelled; steps 6 and 7 are optional.

## Later, unscheduled (handoff S12, S14–S16)
A trends view, more clips plus a second kick type, and a 3D lifting experiment. (The small user test moved into Phase 11.)
