# Decisions

Each entry records the decision, the alternatives, the evidence, the consequence, and a short plain-language summary (*In short*). The architecture decisions D1–D17 in `docs/handoff/03_ARCHITECTURE_AND_DECISIONS.md` stay in force unless an entry here changes them. Add new entries as DEC-NNN.

---

### DEC-001: The handoff wins over the July 2026 plan; conflicts are listed, not merged (Phase 0)
- **Decision:** scope and sequencing follow the handoff (evaluation first, no pro archetypes, no hip rotation). Each conflict is listed in `docs/V1_AUDIT.md` §3.
- **Alternatives:**
  - follow the July plan (ship the app first, evaluate in Phase 3);
  - merge the two plans silently.
- **Evidence:** the audit (V1_AUDIT §2) shows v1's measurements are systematically wrong (letterbox distortion) and its reference profiles are unsourced. Building an app on top of that first would polish wrong numbers.
- **Consequence:** the React PWA, accounts and Postgres are out of scope. The deployed demo (Phase 7) reuses the v1 page.
- **In short:** "I audited before building. The audit showed the measurements were biased, so validating them came before any product work."

### DEC-002: The LLM components are required in the resume version (Phase 0)
- **Decision:** E4 (frontier baseline), S8 (norm extraction + E5) and S9 (structured feedback + validator) are required. The handoff's cut priority (S0–S7, S10, then S9 template-only, then S8) is dropped.
- **Alternatives:** keep the handoff's priority, which protects the measurement work and cuts the LLM work first.
- **Evidence:** the project's purpose is to show evaluated AI engineering, so the LLM work can't be the first thing cut. The handoff's order was written for a one-day time box that no longer applies (one phase per chat).
- **Consequence:** Phase 5 is required; nothing in Phase 6 ships without it.
- **In short:** "The LLM does what LLMs are good at, writing and extracting, under a deterministic validator. Instead of trusting it to measure, I evaluated it head-to-head as a measurement baseline (E4)."

### DEC-003: Only a free-tier Gemini key is available (Phase 0)
- **Decision:**
  - all LLM calls go through one client with a rate limiter and an on-disk response cache;
  - development and feedback calls use a lite model;
  - E4 uses the strongest free-tier model that takes video, within its daily limit;
  - check the current model names and limits in the official docs before use.
- **Alternatives:** paid tier; Claude or GPT as extra baselines. No keys are available for those.
- **Evidence:** project constraint (2026-10-27): free tier only, within its limits; a lite model is acceptable.
- **Consequence:**
  - E4 is single-provider. The README must say so, and must name the exact model; if it's a lite model, it must not be called "frontier";
  - cached responses are committed so a fresh clone reproduces E4 without calls.
- **In short:** "Every LLM call is cached and rate-limited, so the evaluation is reproducible and costs nothing to rerun."

### DEC-004: Tooling is uv + pyproject on Python 3.12, with pytest and ruff (Phase 0)
- **Decision:** `pyproject.toml` with uv and a committed `uv.lock`, Python `>=3.12,<3.13`. Runtime dependencies are added by the phase that first needs them.
- **Alternatives:** a `requirements.txt` (an open question in the July plan).
- **Evidence:** uv 0.13.1 and Python 3.12.6 are installed on the development machine (Apple M2). The pinned minor version avoids wheel gaps in the pose libraries.
- **Consequence:** `uv sync` → `uv run pytest`. There is no ffmpeg CLI on the machine, so Phase 1 prefers PyAV (whose wheels bundle FFmpeg) or installs ffmpeg deliberately.

### DEC-005: New repo; commits are made by hand (Phase 0)
- **Decision:** KickcoachV2 is a new repo. Claude never runs `git commit` or `git push`. At the end of each chat Claude gives explicit `git add <paths>` and `git commit` commands.
- **Alternatives:** Claude commits; reuse the `kickcoach-2` repo.
- **Consequence:** `git add` always lists paths, so private files (gitignored anyway) are never staged.

### DEC-006: Coordinate and angle conventions; every adapter maps through a tested transform (Phase 0)
- **Decision:**
  - **Canonical pose:** `array[T, K, 3]` of `(x, y, conf)` in source-frame pixels, with x right and y down. (v1 used `(y, x)`.)
  - **Joint angles:** interior angle at the middle point, in [0, 180]; a knee angle of 180° is a straight leg.
  - **Trunk lean:** the hip-mid → shoulder-mid angle from image vertical, **positive = forward in the kicking direction**, given an explicit `facing` of ±1. Assumes a level camera.
  - **Degenerate geometry:** gives NaN, never a number.
  - Any model whose input is resized must map back through an explicit, unit-tested transform (`kickcoach/pose/letterbox.py`).
- **Alternatives:** sign the trunk lean by image left/right (v1), which flips for left-facing players.
- **Evidence:** V1_AUDIT §2.2 (letterbox distortion) and §2.7 (sign follows the image, not the kick).
- **Consequence:** ground-truth angles are computed from labels with this same module, so the pipeline and the ground truth can't drift apart.
- **In short:** "A test with known geometry would have caught v1's biggest bug, so every coordinate transform now has one."

### DEC-007: Which ground-truth fields belong to the human (Phase 0)
- **Decision:** the following are ground truth and are entered only by Aamir, in the labelling tool:
  - per frame: keypoints and the ball centre;
  - per clip: contact frame, max-backswing frame, kicking leg, `view_ok_human`.
  - Claude may pre-fill keypoints from a model and show QA flags, but never sets or edits a confirmed value.
  - `level_guess` and `notes` in the manifest are Claude's, and are named as such.
- **Alternatives:** Claude fills `kicking_leg` / `view_ok_human` while curating the manifest. Rejected: E8 scores the quality gate against `view_ok_human`, so filling it in would make E8 circular.
- **Consequence:** the manifest has these columns blank until the labelling homework is done.

### DEC-008: Phase 2 scope is a live demo, RAG + LLM judge, a tool-calling chat, and SportsPose last (Phase 0)
- **Decision:** Phases 7–10 in `docs/PLAN.md`, in that order. SportsPose comes last because it needs an access request and is the most complex.
- **Alternatives:** stop at the resume version.
- **Change to D15** ("no agents"): it still holds for the analysis pipeline. The coach chat is a separate interface where the user's questions are open-ended and need different data each time, which is exactly when tool calling is justified.
- **In short:** "I used tool calling only where the control flow depends on the user's question. The measurement pipeline stays linear and deterministic."

### DEC-009: The clip set is mostly university side-view kick videos, with 3 rear-view twins for the gate (Phase 1)
- **Decision:**
  - 26 clips from 25 YouTube videos: 23 side-on kicks (`purpose=eval`) and 3 rear views of kicks that are also in the set side-on (`purpose=gate_only`, for E8; no keypoints needed).
  - Held out: c03, c11, c12, c15, c16, drawn with a fixed seed over groups of clips that share a kicker or a kick, so no person appears in both splits (`tools/make_manifest.py`).
  - Every candidate considered, kept or rejected, with the reason, is in `data/candidates.csv`.
- **Alternatives:**
  - the 12 hackathon links as the core: 9 of them are broadcast compilations, long tutorials or curled inside-foot free kicks, so only 3 were usable;
  - side-on clips only: E8's gate precision/recall would then have no negatives to score.
- **Evidence:**
  - YouTube storyboards (about 1 frame per second, previews only) were checked by eye before anything was downloaded; Aamir approved 29 clips from the contact sheet on 2026-10-27.
  - After download, c21, c22 and c29 (one University of Hertfordshire video, 3 cameras) were dropped: every camera freezes into a drawn skeleton overlay at the backswing, so ball contact is only visible under drawn joint markers, which would leak positions to the pose models and to Gemini.
- **Consequence:**
  - The set leans towards indoor university-lab footage of student kickers (17 of 23: c01–c08, c10–c18); 2 pro clips (c19, c20); 5 outdoor clips (c09, c19, c23, c24, c25). The README must say so.
  - `level_guess`, `slow_motion` and the windows are Claude's curation, not ground truth (DEC-007); `kicking_leg` and `view_ok_human` stay blank for Aamir.
- **In short:** "I curated from previews, documented every rejection, and kept a seeded, person-disjoint held-out split. I also added deliberately bad-view clips, because a quality gate can't be evaluated without negatives."

### DEC-010: Velocities are trusted only when real time per frame is known (Phase 1)
- **Decision:** `kickcoach/ingest/fps.py` marks a clip `fps_trusted` only if its frame timestamps are evenly spaced, it shows no sign of repeated frames, and either the source states the capture rate or the clip is real time. Untrusted clips still get angles; velocities are withheld.
- **Evidence:**
  - All 26 streams are constant-frame-rate (PyAV `r_frame_rate` = `avg_frame_rate` within 0.2%).
  - 3 of 26 are trusted: c13 (300 fps stated), c19 (240 fps stated), c25 (real time: its burned-in camcorder timecode advances one frame per video frame).
  - 18 other side-on clips and the 3 rear twins are slow motion with no stated capture rate (stated as slow by the source, or visibly slowed); c23/c24 can't be told apart from real time.
  - The first repeated-frame heuristic counted any still frame pair and flagged c21 at 80% because the player stands still for 100 frames. It was replaced by one that only counts short stalls bounded by motion; a synthetic clip with every frame shown twice scores above 0.4, and the real clips score 0.00.
- **Consequence:** velocity features will be `not_measured` on most clips, so E8's "velocities correctly withheld" has real cases. Both repeat thresholds are marked `heuristic` in code.
- **In short:** "A 240 fps clip played at 30 fps has 1/240 s between frames, not 1/30. The container can't tell you that, so the pipeline refuses to compute speeds unless the real time per frame is established."

### DEC-011: Three pose adapters into one 23-point skeleton (Phase 1)
- **Decision:**
  - The canonical skeleton is COCO-17 plus big toe, small toe and heel per foot (`kickcoach/pose/skeleton.py`). Missing points are (nan, nan, 0).
  - Models:
    - **MoveNet:** v1's own `3.tflite` (SinglePose Lightning), whole frame, run on `ai-edge-litert`, with an exact integer-pixel letterbox inverse (`PixelLetterbox`).
    - **RTMPose-m Halpe26:** through rtmlib `BodyWithFeet` (balanced), on a YOLOX-m person box chosen by a largest-then-IoU tracker.
    - **MediaPipe Pose Landmarker heavy:** VIDEO mode; FOOT_INDEX is mapped to big toe, and there is no small toe.
  - Outputs are cached as `data/cache/pose/<model>/<clip>.npy` with a JSON sidecar (weights hash, timing).
- **Alternatives:**
  - full TensorFlow for TFLite (heavy; LiteRT has an arm64 wheel);
  - RTMPose-x (≈2–3× slower on this CPU; revisit if E1 is close);
  - YOLO11-pose or ViTPose (they need torch and have no foot points).
- **Evidence:**
  - All three overlaid on a 1080p frame sit on the joints.
  - MediaPipe's "NORM_RECT without IMAGE_DIMENSIONS" warning was checked visually and is harmless here.
  - Speeds on the M2 CPU: RTMPose 2.5–3.4 frames/s, MediaPipe ≈11.
- **Consequence:** confidences are not comparable across models (score vs SimCC peak vs visibility), so E1 compares positions, not confidences. Model weights came from rtmlib's OpenMMLab zoo and Google's MediaPipe bucket (gitignored, hashes recorded).
- **In short:** "Every model goes through one tested adapter into one skeleton, so E1 compares like with like. The adapters are also where v1's letterbox bug lived, so each has a known-geometry test."

### DEC-012: Gemini setup for E4: gemini-3.8-flash, 24 fps sampling, inline video, cached (Phase 1)
- **Decision:**
  - **E4 model:** `gemini-3.8-flash`, the strongest model the pricing page lists as free (2026-10-28). `gemini-4-argon` and `gemini-3.1-pro-preview` are paid-only.
  - **Dev and feedback model:** `gemini-3.5-flash-lite`.
  - **Request:** video sent inline at `VideoMetadata.fps = min(playback, 24)` (24 is the API maximum; the default of 1 fps would miss the kick), temperature 0, JSON schema output.
  - **Client:** one client (`kickcoach/llm/gemini.py`) with an on-disk cache keyed on model + prompt + schema + video hash + fps, ≥12 s between calls across processes, and backoff on 429/503.
- **Evidence:**
  - Free-tier RPM/RPD are no longer published in the docs (they point to AI Studio), so the 12 s spacing is a heuristic.
  - The first gut-check runs hit 503 "high demand" and one dropped connection, which is why the client backs off and retries.
- **Consequence:**
  - E4 is single-provider and uses a Flash model. The README must name `gemini-3.8-flash` and not call it a frontier Pro model.
  - At 24 fps a 30 fps clip loses 1 frame in 5, which limits Gemini's contact-time precision (state it in E4).
  - The gut-check (3 clips) returned numbers for every field with no "cannot determine". Video tokens per sampled frame were 63–66 (c10 6,804 / 108; c19 9,072 / 144; c25 10,296 / 156), close to the docs' low resolution (66). Phase 4 must decide, and report, whether E4 sets `media_resolution` HIGH (≈258 per frame, about 4× the tokens).
- **In short:** "The baseline gets every advantage I can give it within the free tier: as many frames as the API allows, the exact angle definitions, and an explicit 'cannot determine' option."

### DEC-013: Labelling tool and guide: a local browser tool, COCO visibility flags, joint centres from ISB, pre-fill kept for an anchoring check (Phase 2)
- **Decision:**
  - **Tool** (`tools/label_tool.py` + `tools/label_ui/`): a stdlib `http.server` on 127.0.0.1 serving a vanilla-JS canvas page. It adds no dependencies. Every change autosaves `data/labels/<clip>.json` through a validating, atomic write (`kickcoach/labels/schema.py`).
  - **Points:** 12 body points (shoulder, hip, knee, ankle, heel and big toe, left and right) plus the ball centre. Small toes are not labelled: from the side, a shoe hides which toe is which.
  - **Visibility:** COCO's flags (0 not labelled, 1 labelled but not visible, 2 visible). Unknown points have no coordinates, so nothing is guessed.
  - **Clip fields:** the planned ones plus two more:
    - `kick_direction` (image left or right), needed to sign trunk lean under DEC-006;
    - `contact_between_frames`, for when no frame shows the touch.
  - **Landmark definitions** (`docs/LABELLING_GUIDE.md`), each cited:
    - joint centres from ISB (Wu et al. 2002, 2005);
    - where those centres appear in a side view, from a validated 2D method's surface landmarks (Fernández-González et al. 2020);
    - foot points from OpenPose / COCO-WholeBody;
    - ball centre and digitised landmarks as in Nunome & Ikegami 2006;
    - event definitions from Kellis & Katis 2007.
    - The one uncited step, the side-view shoulder position, is marked as our reading.
  - **Pre-fill:** RTMPose. The tool stores the pre-fill next to every label.
  - **QA flags** (`kickcoach/labels/qa.py`), shown and never applied:
    - a missing required point;
    - a point more than 0.5 thigh lengths from the model (*heuristic*);
    - a left/right pair swapped relative to the model;
    - a thigh or shank length that varies by more than 20% across frames (*heuristic*);
    - event order;
    - incomplete fields.
  - **Timing:** active seconds (input within the last 30 s, *heuristic*) and on-screen seconds, per frame and per clip's events.
  - **Repeat mode:** a seeded, round-robin sample of 10 confirmed frames that are at least 48 h old. The server refuses to send the first labels (HTTP 403) while it runs.
- **Alternatives:**
  - a matplotlib tool: dragging, zooming and keyboard handling vary across macOS backends;
  - CVAT or Label Studio: heavy installs for 2 frames × 23 clips;
  - pre-fill from a random model per clip: only RTMPose has every foot point.
- **Evidence:**
  - 28 tests in `tests/test_labels.py`, including an HTTP round trip on a temporary data folder and repeat mode never sending labels;
  - headless-browser screenshots of both views;
  - a labelling trial on c10 and c01, in-session.
  - From the trial: arrow keys, the scrubber and play now browse the whole clip from the keypoint screen, with the labelled points drawn faint; the notes box is bigger.
- **Consequence:**
  - Pre-filling anchors the labeller towards RTMPose, which is one of the models E1 compares. E1 must report how far labels moved from the pre-fill. In the trial, 12 of 72 body points were left exactly at RTMPose's guess.
  - In the trial, 0 of 78 points were marked occluded, including the far-side hip, which in a side view is behind the near hip. The guide now states the near-side rule and when the far-side hip and shoulder count as visible.
- **In short:** "Pre-filling from a model saves labelling time, but it anchors the labeller to the very model being evaluated. So the tool keeps the pre-fill next to every label, and the evaluation reports how far the human moved it."

### DEC-014: Labelling budget: events on every clip, keypoints on max backswing and contact; contact − 1 optional (Phase 2)
- **Decision:**
  - **Required homework:**
    - the clip events for all 26 clips: kicking leg, ball direction, view OK, contact, max backswing. The 3 `gate_only` clips need only view OK;
    - keypoints on the **max-backswing and contact** frames.
  - **Target:** keypoints on all 23 eval clips.
  - **Minimum:** 15 clips, including all 5 held-out ones (c03, c11, c12, c15, c16).
  - **Contact − 1 is optional.** It's added back only if, after the first 5 homework clips, `--report` shows a median of 90 s or less per frame (*heuristic*). The tool shows it as "(optional)".
- **Evidence** (`tools/label_tool.py --report`, `results/labelling/timing_*.csv`; trial on 2 clips, active time):
  - per frame: median 152 s, range 77–292 s. c10 fell from 292 → 135 → 77 s as the labeller learned the tool. c01 rose from 100 → 170 → 240 s, because the dark legs overlap near contact (the labeller's note);
  - clip events: 235 s (c10) and 283 s (c01).
  - Projections at these medians, for the 21 eval clips still to do:
    - 23 × 3 frames: 251 min;
    - 23 × 2: 197 min;
    - 15 × 2 (events on all 23): 157 min.
  - The plan had assumed 1–2 h.
- **Alternatives:**
  - 23 × 3 (about 4 h more);
  - fewer clips with 3 frames each: fewer clips weakens E2/E3/E4, which are per-clip comparisons, more than a missing contact − 1 weakens E1;
  - fewer points per frame: no, because E1 and E2 need all of them.
- **Why these frames:**
  - Contact and max backswing feed E2 and E4 directly (knee angle at contact, max knee flexion, trunk lean).
  - Contact − 1 only adds E1 frames.
  - Events are cheap per unit of evaluation value: E3 and E8 use every clip.
- **Consequence:** about 2.6–3.3 h of labelling still to do, in sittings of no more than about an hour. Re-run `--report` after the first sitting to re-estimate. The README reports the n that results.
- **In short:** "I timed a pilot before committing to a labelling budget. It came out at about twice my estimate, so I cut the least valuable frame type rather than the number of clips, because the per-clip comparisons need n more than E1 needs a third frame."

### DEC-015: Contact is the first real touch, not the first overlap (2026-10-30)
- **Context:** the guide defined contact as the first frame with "no gap" between shoe and ball, and said contact − 1 has a gap. Labelling showed that two frames in a row usually show no gap. In the first, the shoe only overlaps the ball's outline in the picture; in the second, the shoe is against the ball. A frame with a gap is hardly ever followed directly by a real touch.
- **Decision:**
  - contact = the first frame in which the shoe is pressed against the back of the ball. Signs: the ball looks squashed or pushed, or it has moved by the next frame;
  - an overlap-only frame before it is contact − 1;
  - contact − 1 is just the frame before contact and doesn't need a gap;
  - the "no frame shows the touch" fallback is unchanged.
  
  This is our convention, not a cited one: the literature's "ball impact" comes from high-speed video, where the ambiguity is much smaller.
- **Why:** "no gap" is a 2D cue. A shoe in front of the ball, or beside it in depth, closes the gap before any touch. The refined rule matches how the 5 held-out clips were already labelled, so their labels stay consistent.
- **Consequence:** E2 compares the detected contact frame with this definition. At these frame rates, a disagreement of one frame between reasonable labellers is expected, so E2 reports frame offsets, not only exact matches.
- **In short:** "My first contact definition used a 2D cue, the gap closing, that fires a frame early, because overlap in the picture isn't touch in depth. The labeller caught it from the frames, so we redefined contact by its physical signs and kept it consistent with the labels already made."

### DEC-016: The pose model is RTMPose-m Halpe26, chosen by E1 (Phase 3)
- **Decision:** the pipeline uses RTMPose-m (rtmlib `BodyWithFeet`, balanced). RTMPose-x is not run: E1 wasn't close.
- **Rule, fixed before the results were seen** (docstring of `eval/e1_pose.py`): the best PCK@0.1 on the core points (shoulders, hips, knees, ankles) of the required frames wins, if the clip-bootstrap 95% CI of its lead excludes 0 *and* it also leads on the points the labeller moved. Otherwise Aamir picks.
- **Evidence** (`results/e1_summary.md`; 19 clips, 38 required frames, 304 core points):
  - core PCK@0.1: **RTMPose 85% (CI 81–89%)**, MediaPipe 65% (60–72%), MoveNet 22% (17–27%);
  - lead over MediaPipe: +19.7 points (CI +13.5 to +25.7, 2,000 clip resamples);
  - on moved points only, where pre-filling can't help RTMPose: 73% vs 54% vs 20%;
  - held-out clips: 89% vs 69% vs 25%;
  - raw knee angle at contact, from each model's points vs the labels' points: MAE 2.4° / 5.5° / 10.8°.
- **Alternatives:** MediaPipe (faster, ≈11 vs 2–3 frames/s on the M2, but places hips 0.08 thigh lengths low and is 20 PCK points worse); MoveNet (v1's model: no foot points, and 7 of 43 frames have the whole legs swapped).
- **Consequence:** the 37% of points left at the pre-fill flatter RTMPose's absolute numbers, so E1 reports moved-only numbers next to them. The README must say which model was evaluated and that the labels were pre-filled from it.
- **In short:** "I fixed the selection rule before looking, and because the labels were pre-filled from one of the candidates, I also scored the models only on points the labeller had moved. RTMPose won both ways, by about 20 points."

### DEC-017: `ankle_angle_contact` is dropped (Phase 3)
- **Decision:** the feature is removed from the pipeline and the output. Approved by the author on 2026-10-30 (dropping a feature needs that approval).
- **Evidence (E1):**
  - only 16% (RTMPose) and 21% (MediaPipe) of the kicking big-toe labels at contact are within 0.1 thigh lengths, though Aamir marked all 19 as visible;
  - the knee–ankle–toe angle at contact comes out 28° (RTMPose) and 21° (MediaPipe) too small on average.
  - JOURNAL Phase 2 §22 explains why: the toe at impact is blurred and often behind the ball.
- **Alternatives:** always report it as `not_measured` (a feature that never yields a value); report it with a caveat (a 28° bias makes any "locked ankle" verdict meaningless).
- **Consequence:** the handoff's S5 cut line allows exactly this. The README's limitations list it, with these numbers.
- **In short:** "The ankle angle failed E1 for every model: the toe at impact is blurred or hidden behind the ball. So I cut the feature rather than ship a number with a 28-degree bias."

### DEC-018: The ball detector is YOLOX-m COCO via onnxruntime, with our own decode (Phase 3)
- **Decision:**
  - Megvii's `yolox_m.onnx` (0.1.1rc0, Apache-2.0, 101,259,744 bytes, sha256 `21ff6cfd…`) in `models/yolox/` (gitignored); it's the same file rtmlib's Animal solution uses;
  - rtmlib's YOLOX preprocess and inference, with our own decode, because rtmlib's drops the scores and keeps only boxes ≥ 0.7;
  - "sports ball" candidates are kept from score 0.05 (top 5 per frame), cached in `data/cache/ball/`.
- **Alternatives:**
  - ultralytics YOLO (needs torch, AGPL);
  - YOLOX-s (36 MB, more misses);
  - no detector, with contact from foot speed only. That would also drop `plant_foot_offset` and E3's comparison.
- **Evidence:** 3.6 frames/s on the M2 (all 26 clips ≈ 20 min). On c03 the resting ball scored 0.9, with its centre within 1 px of Aamir's ball label. c06's ball is never found (4 of 150 frames, best 0.22): the player is small in frame.
- **In short:** "I reused the detector runtime I already had, ONNX through onnxruntime, so no torch, but wrote my own decoding. The library's version threw away the confidence scores, and a blurred ball needs low-confidence candidates plus tracking logic rather than a hard 0.7 cut."

### DEC-019: Signal stage: swap fix by Viterbi, a segment-length glitch mask, an 18 Hz cutoff, and a break at contact (Phase 3)
- **Order:** mask low confidence (< 0.3, *heuristic*) → fix leg swaps → mask segment-length glitches → fill gaps of ≤ 3 frames (*heuristic*) → zero-lag Butterworth (order 2, forward-backward). The plan listed the gap fill before the swap fix; fixing swaps first means no gap is ever bridged across a swap.
- **Swap fix:**
  - a two-state Viterbi over "as the model says / swapped" for the leg chain (knee down);
  - motion cost = distance from the constant-velocity prediction, in thigh lengths, capped at 1.0 per frame (*heuristic*);
  - a prior of 0.5 per overruled frame (*heuristic*);
  - hips and shoulders are not swap-fixed.
- **Glitch mask:** a leg chain is masked on frames where its thigh or shank is more than 1.5× longer or shorter than its ±7-frame rolling median (*heuristic*; 3.1% of frames).
- **Cutoff:**
  - with a trusted frame rate: **18 Hz**, the top of the 6–18 Hz that kicking studies used for swing kinematics (Nunome et al. 2006, p. 11). Above Nyquist (real-time 30 fps) there is no filtering;
  - otherwise **0.15 cycles/frame** (*heuristic*): 18 Hz at 120 fps. On the tuning split all cutoffs scored within ~0.4° of each other (`eval/tune_cutoff.py`).
  - Single-pass design is corrected so the two-pass response is −3 dB at the target (derived, then checked with `sosfreqz`).
- **Break at contact:** each side of the contact frame is filtered and differentiated separately, because filtering through impact distorts the kinematics (Nunome et al. 2006, p. 12). On the synthetic kick it recovers the 1000°/s peak within 5%, and filtering through impact does worse.
- **Evidence:**
  - the first swap fixer (greedy, frame to frame) swapped legs RTMPose had right: c06 frame 64 went to +45° knee error, and 30–59-frame stretches flipped;
  - the Viterbi version harms no labelled RTMPose or MediaPipe frame, and catches RTMPose's real swaps in kick windows (c07 frame 53, c15 frames 65–67; checked visually);
  - it does *not* repair MoveNet's swaps, which are mostly 14–32-frame runs;
  - a hip-swap test on the synthetic kick (hips 4 px apart, 2 px noise) invented a swapped stretch, so hips were removed from the fix;
  - the glitch mask fixed c12's backswing (a 2.9° "knee angle" at frame 44) and c07's (a 137° spike from a partial swap).
- **Limitation:** partial swaps, with the knee on one leg and the ankle on the other (c07 frames 59 and 67; c14 frames 71–74), are only caught when they also break a segment length.
- **In short:** "My first leg-swap fix made the data worse: in a kick the foot moves as far in one frame as the gap between the legs, so nearest-neighbour matching flipped correct legs. I replaced it with a Viterbi pass that predicts each foot from its own velocity and has to pay to overrule the model, then checked it on frames I could see."

### DEC-020: Events are found from the ball first, then the kicking leg (Phase 3)
- **Contact**, in order of preference:
  1. `ball_motion`: last frame at rest, with the ball seen moved on the next frame. "At rest" = within 0.1 ball diameters of the rest spot (*heuristic*, from tuning clips: resting jitter ≤ 0.02 D, first moved frame 0.15–0.27 D).
  2. `ball_bracket`: the touch is hidden between the last frame at rest and the first frame moved. Inside that bracket, the frame where one foot is fastest relative to the other.
  3. `foot_speed`: no ball. Same rule over the whole clip, reported as low confidence.
- **Kicking leg:** the foot with the larger net displacement up to contact, over the shortest look-back in which one foot has moved half a thigh length (*heuristic*). Kick direction is the sign of that foot's horizontal velocity.
- **Max backswing:** the deepest knee flexion in the flexion valley that ends at contact. The search stops where the knee is back to its contact angle, once the valley is 20° deep (*heuristic*). Our convention: the literature defines the frame, not the search.
- **Evidence** (`results/pipeline_summary.csv`; a preview, not E3):
  - the whole-clip peak foot speed picked the wrong leg on 7 of 19 labelled clips, because the peaks were follow-through glitches 30–60 frames after contact;
  - the speed difference between feet also failed (offsets −66 to +68 frames), because running strides have the same signature;
  - ball first: kicking leg 19/19; contact exact on 11/19 and within 2 frames on 17/19 (c13 −4; c06, with no ball, +17);
  - inside brackets, the peak runs 0–3 frames early (median −2) on the tuning clips, even unfiltered.
- **Open for Phase 4:** whether to correct the bracket's −2-frame lead. E3 decides, with the held-out clips.
- **In short:** "Pose alone can't tell when the kick happens, because a running stride looks the same: one foot swinging, one planted. The ball can, and finding contact from the ball doesn't need to know which leg kicked. So I find contact first, then name the leg as the foot that moved into the ball."

### DEC-021: Features, sanity bounds and the quality gate (Phase 3)
- **Features:** `knee_angle_contact`, `max_knee_flexion_backswing`, `trunk_lean_contact`, `plant_foot_offset` (plant ankle to the resting ball, along the kick, ÷ plant leg length), `segment_sequence_order` and `peak_knee_ext_velocity`.
- **Velocity gating:**
  - velocities, and the thigh-to-shank lag in ms, need a trusted rate of at least 120 fps (handoff D6). c25 (real time, 30 fps) gets neither;
  - the order of the peaks is reported for any evenly timed clip.
- **`keypoint_score`:** the model's own mean score for the points used, not a calibrated probability.
- **`contact_confidence`:** a category (high/medium/low) from a stated rule, not an invented number.
- **Bounds** (`kickcoach/features/bounds.py`):
  - knee extension ≤ 1.5 × 1720°/s and shank ≤ 1.5 × 1891°/s (Kellis & Katis 2007, p. 155; the 1.5 margin is *heuristic*);
  - knee angle 30–180°, trunk lean ±60° and plant offset ±1 leg length are *heuristic*, each with its reasoning in code.
  - A value outside its bound becomes `not_measured` with the reason, never clamped.
- **Gate** (`kickcoach/quality/gate.py`): refuses with reasons and no features when:
  - shoulder width / torso length in the kick window is > 0.5 (*heuristic*);
  - the median keypoint score is < 0.4 (*heuristic*);
  - or, for a foot-speed contact only, the kicking foot moved < 1.5× the plant foot.
  - A missing ball and untrusted timing are warnings that withhold the affected features, not refusals.
- **Evidence:**
  - on whole clips the view ratio overlapped (c23, not OK: 0.519; c01, OK: 0.527);
  - in the kick window it separates: side-on ≤ 0.437, not-OK views ≥ 0.564. The threshold was set on the tuning split, which holds all 4 negatives, so E8's tuning numbers are optimistic and its held-out numbers can only show false refusals;
  - on all 26 clips: 88 measured numeric values, 0 outside a bound; 4 refusals (c23, c26–c28), each with a reason and no features;
  - velocities on the two trusted high-speed clips: c13 1147°/s and c19 1698°/s, both inside Kellis & Katis's 860–1720°/s.
- **In short:** "A quality gate only means something if you can measure it against labels, so I calibrated it on the tuning clips and said so. And a missing ball doesn't reject the whole clip: it withholds exactly the one feature that needs the ball."

### DEC-022: The ball-bracket contact pick is moved 2 frames later, inside the bracket (Phase 4, E3)
- **Context:** DEC-020 left open whether to correct the bracket's lead. When the ball is hidden at the touch, contact is the frame inside [last seen at rest, first seen moved − 1] where one foot is fastest relative to the other, and that ran early.
- **Decision:** `BRACKET_LEAD_FRAMES = 2` (`kickcoach/events/__init__.py`): the pick moves 2 frames later, clamped to the bracket's last frame. Ball-motion contacts and the foot-speed fallback are unchanged.
- **Rule, written in `eval/e3_contact.py` before the corrected numbers were computed:** the shift = minus the median offset of the tuning bracket clips. Adopt it if leave-one-clip-out on those clips lowers the mean absolute error, and the held-out bracket clips don't get worse. Phase 3's preview had already shown the uncorrected offsets, so the rule wasn't blind to them.
- **Evidence** (`results/e3_summary.md`):
  - tuning bracket clips (7): early on 6, late on none (offsets 0, −1, −2, −2, −2, −2, −4);
  - leave-one-out: mean absolute error 1.86 → 0.57 frames (every left-out median gives +2);
  - held-out: the one bracket clip (c15, +1) has a 3-frame bracket, so the clamp leaves it unchanged. The held-out set could only have shown harm, and it shows none;
  - why it matters (E2): two frames early, the knee is 10–15° more bent on c02, c05, c13, c14 and c17. v2's knee-at-contact MAE went 4.8° → 3.4° and its bias −3.5° → +0.2°, all on tuning clips (in-sample; the held-out MAE stays 1.2°);
  - cost: c01, whose uncorrected pick was exact, is now +2 frames and +16.5° (`results/failures.md`).
- **Why the lead exists:** the foot's speed peaks just before it is pressed into the ball. The central difference at the contact frame already includes the post-impact slowdown, and the filter spreads it back further.
- **Alternatives:** no correction (DEC-020's state); a time-based shift (needs a trusted frame rate, which most clips lack); always taking the bracket's last frame (wrong when the ball stays hidden long after the touch: c13's bracket is 42 frames).
- **Limitation:** the shift is in frames, calibrated on slow-motion clips (most of unknown factor). On real-time video, 2 frames is about 67 ms. The clamp limits the damage there, because the ball reappears quickly, so brackets are short.
- **In short:** "The ball told me *which interval* contact was in, and foot speed picked the frame inside it, but consistently two frames early. I wrote the adoption rule down before computing it: leave-one-out on the tuning clips, and no harm on held-out. It cut the contact error by two-thirds, and I kept the shift inside the ball's bracket so it can't run away on real-time video."

### DEC-023: E4 asks Gemini at HIGH media resolution and HIGH thinking, with no temperature (Phase 4; revises DEC-012)
- **Decision** (`eval/e4_gemini.py`, prompt `e4-v2`):
  - `media_resolution` HIGH;
  - `thinking_level` HIGH;
  - no temperature;
  - prompt v2 adds the max-backswing time and knee angle, so Gemini answers every angle E2 scores;
  - the labelled kicking leg is given, as before;
  - 5 clips are asked twice, to measure how much an answer varies between identical calls;
  - requests time out after 15 min and are retried like a 503.
- **Evidence (docs checked 2026-10-31):**
  - **Media resolution:** Google's media-resolution page puts video at 70 tokens a frame for low/medium and 280 for high on Gemini 3/4 models, and calls high "required only when the use case involves … small details within video frames". A knee angle on a player who fills a fifth of a 480×360 frame is a small detail. Measured: c10's 108 sampled frames took 28,728 video tokens (266 a frame) vs 63–66 in the Phase 1 gut-check.
  - **Thinking level:** the Gemini 3.8 Flash developer guide (Google Cloud docs) gives MEDIUM as the default, and recommends HIGH for "split-second movement detection (sports, editing)".
  - **Temperature:** the same guide says temperature, top_p and top_k "are ignored by the backend". So DEC-012's "temperature 0" never applied, and the Phase 1 gut-check wasn't deterministic either.
- **Cost:** 25–34k thinking tokens and 6–8 min per call; one call hung for over 30 min with the SDK's default of no timeout. Three workers ran in parallel, each request still spaced ≥ 12 s apart by the shared rate file.
- **Free-tier limit, found the hard way:** **20 requests a day per model** for `gemini-3.8-flash`, from the 429's QuotaFailure (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 20) on 2026-10-31; it resets at midnight Pacific (12:30 IST). Hung, timed-out and 503'd calls appear to count: only 10 answers came back before the quota of 20 ran out (an inference; Google doesn't itemise it).
  - E4 needs 32 requests (19 scored clips + 6 repeats + 7 unlabelled clips). Day 1 got 10 answers.
  - `collect` now runs the requests in priority order (scored clips, repeats, the rest), skips cached ones, and stops at the daily quota (`DailyQuotaExhausted`), so `uv run python -m eval.e4_gemini collect` once a day finishes it in two more days.
- **Preliminary result (8 of 19 scored clips; `results/e4_summary.md`):**
  - **Contact timing is good:** within ±1 frame on 6 of 7 answers.
  - **Angles are not:** knee at contact +22.9° bias (MAE 22.9° vs the pipeline's 2.3° on the same clips), max flexion MAE 19.3°, trunk 10.2°. Its knee values (138–161°) don't track the labels (r −0.42; pipeline +0.88).
  - **Confident when it shouldn't be:** it gave all 4 numbers on both not-side-on clips the pipeline refuses (c23, c26).
  - **Odd failure:** on c01 it said the video was a single frozen frame (it isn't).
- **Full result (2026-11-02, Phase 5 chat; all 19 scored clips asked, 18 answered):**
  - **Knee at contact:** Gemini 18.5° MAE / +18.4° bias vs the pipeline's 2.7° on the same 18 clips (held-out: 19.3° vs 1.2°). Its values track the labels weakly (r +0.27; pipeline +0.93). On the 13 clips where its contact was within ±1 frame, the knee error is still +21.0°.
  - **Max knee flexion:** 15.2° vs 4.9°. **Trunk lean:** 8.8° vs 2.6°.
  - **Timing:** contact ±1 frame on 72% of answers, ±2 on 83% (pipeline 84% / 95%); backswing ±2 on 22% (pipeline 58%).
  - **Same question twice** (5 comparable pairs): answers moved by up to 19.4° (knee), 19.6° (max flexion) and 15.8° (trunk); contact frames by 0–1.
  - **c01:** the first call said the video was "a single frozen frame"; the repeat gave numbers and explained them with "edited pauses". Neither is true: ingest's frame differences never drop below 0.66% of pixels changed (the still threshold is 0.01%). So it's the model confabulating a property of the video, not a file to re-encode.
  - **Not side-on:** still all four numbers on c23 and c26.
  - **Quota:** 17 answers + 3 × 503 = 20 requests before the daily 429, consistent with 503s counting against the quota.
  - **Left:** the 5 unlabelled clips (refusal behaviour only); `collect` asks them on any later day.
- **Consequence:** the gut-check (e4-v1, LOW, temperature 0) keeps its own cached responses; `tools/gemini_gutcheck.py` is pinned to v1.
- **In short:** "I read the model's current docs before running the baseline, and they changed my setup. The temperature I'd been setting was silently ignored, and the defaults were the low-detail, medium-thinking ones. So I gave Gemini high resolution and high thinking, and I measured its run-to-run variance instead of assuming determinism."

### DEC-024: E2's v1 baseline is v1 as it shipped, replayed, plus the plan's "v1-style" ablation (Phase 4)
- **Decision:** E2 and E8 compare against `eval/v1_replica.py`, not just "MoveNet without filtering". The replay uses:
  - v1's own MoveNet weights on **BGR** frames (V1_AUDIT 2.9);
  - v1's **letterbox-ignoring** mapping (2.2);
  - its causal EMA;
  - its trunk sign by image side (2.7);
  - its knee velocity across gaps (2.4);
  - its "kick" = the highest one-step ankle speed in the clip (2.3, 2.8).

  Keypoints are cached by `tools/run_v1_replica.py` (about 40 s for all clips). One difference from the original: OpenCV area resizing instead of TF's bilinear.
- **And the plan's "v1-style" rows:** E2 also scores, at the labelled frames, a 2×2 of model (MoveNet, RTMPose) × cleaning (raw, `kickcoach.signal`). That separates the model's gain from the cleaning's gain.
- **Why:** the resume template claims an error reduction "versus the hackathon version". A corrected MoveNet without filtering isn't the hackathon version: it fixes v1's two largest bugs for free.
- **Evidence** (`results/e2_summary.md`):
  - **Knee at contact, end to end:** v1 12.8° vs v2 3.4° MAE; paired difference +9.4° (clip-bootstrap 95% CI +4.9 to +14.4). Held-out: 9.6° vs 1.2°.
  - **Trunk:** 16.5° vs 2.7°.
  - **v1's "kick" frame:** never within ±2 frames of contact (it's 13–71 frames off), and it picked the labelled leg on 9 of 19 clips.
  - **At the labelled frames:** MoveNet raw 10.8°, RTMPose raw 2.4°, RTMPose cleaned 2.4°. Nearly all the measurement gain is the model; the cleaning's job is events and velocities.
- **In short:** "I didn't compare against a strawman or a cleaned-up version of the old code. I replayed the hackathon pipeline bug for bug, then ablated model and cleaning separately, so I can say where the 9° came from: mostly the pose model and finding the right frame, not the filter."

### DEC-025: The low-pass cutoff stays at 0.15 cycles/frame after E2's sensitivity check (Phase 4)
- **Decision:** no change to DEC-019's cutoff.
- **Evidence** (`results/e2_summary.md` §4, `results/e2_cutoff.csv`):
  - At the labelled frames, every cutoff from none to 0.3 cpf lands within ~0.5° (knee at contact 2.7° tuning / 1.5° held-out at every cutoff).
  - End to end, the cutoff moves the *events*: c06's foot-speed contact is near-right at 0.05–0.1 cpf and 17 frames late at ≥ 0.15; bracket picks shift by a frame; c01's backswing search fails at 0.2. No single cutoff wins on all three angles (tuning: knee at contact is best at 0.2, max flexion at 0.08–0.1, trunk at 0.08), and the held-out rows are flat.
- **Discovery:** with the break at contact, the contact frame is the last sample of a filtered piece, and `sosfiltfilt`'s odd padding pins it. Its filtered value equals the raw value to ≤ 0.01 px (checked on c02, c07, c10, c13). So contact-frame angles are effectively unfiltered. That doesn't hurt here (raw RTMPose is 2.4° MAE at contact), but it means the filter can't help a noisy contact frame.
- **In short:** "The sensitivity sweep said the filter barely changes the angles at the right frames; it changes *which* frames the event detectors pick. Retuning the cutoff to win on 14 clips would have been overfitting a few event flips, so I left it and reported the sweep."

### DEC-026: Norms come from 4 open-access papers, extracted by `gemini-3.6-flash`, verified by Aamir, with 2 figures-only papers as hallucination controls (Phase 5)
- **Decision** (`tools/extract_norms.py`, prompt `norms-v1`):
  - **Sources:** Kellis & Katis 2007 (review), Scurr & Hall 2009, Kapidžić et al. 2014, Xu et al. 2025 (`knowledge/sources.md`).
  - **Controls:** Nunome & Ikegami 2006 and Harrison & Mannering 2006, whose results are only in figures, so any row from them is invented by construction.
  - **One structured-output call per paper** with `gemini-3.6-flash`: free tier (pricing page, 2026-11-02), and quotas are per model, so E4's and the feedback model's stay untouched.
  - **Schema:** D12 plus `quote`, `condition`, `angle_convention`, `primary_source` and `unit` (`kickcoach/compare/norms.py`).
  - **Deterministic QA flags:** the quote is found verbatim in the text; each number appears in its quote. Flags only; they never change a row.
  - **Files:** `knowledge/norms_candidates.csv` is the raw output, never edited. `knowledge/norms.csv` is the copy Aamir verifies (`review`, `verified`). `knowledge/norms_missed.csv` lists misses.
  - **Caches:** the raw cache holds each paper's full text, so it stays local (`data/cache/norms_extraction/`). `results/norms_extraction/` keeps a redacted record per paper (answer, usage, SHA-256 of the text).
- **Why these papers:** they were the only open-access ones found that print numbers for variables the pipeline measures:
  - Scurr & Hall: knee angle at impact, projected onto the sagittal plane;
  - Xu et al.: peak swing knee flexion;
  - Kellis & Katis Table 4: peak knee extension velocity by study;
  - Kapidžić et al., Scurr & Hall, and Kellis & Katis (citing Hay): support-foot distance.

  No source was found for trunk lean at contact.
- **Dead ends:** the two ISBS papers (figures only; reused as controls). PMC served a CAPTCHA to scripted fetches of Kellis & Katis, so the journal's own PDF was used. johk.pl was unreachable, so Kapidžić et al. came from PMC's BioC text, with tables cited by number.
- **Verification** (by the author, 2026-11-02): 7 table groups with every field shown, all answered "All correct". That includes the 4 points I raised as QA (ages inferred as "adult"/"adolescent"; the Hay rows drop the paper's "not confirmed experimentally"; a duplicate row that the table really contains; Scurr & Hall's method worded as "3D motion capture"). Two values stated in Kellis & Katis's prose (p. 162) had no row. Claude made that call: they're counted as missed and kept out of `norms.csv`, because the text contradicts the paper's own Table 4.
- **E5** (`uv run python -m eval.e5_norms` → `results/e5_summary.md`):
  - 32 rows, 0 hallucinated (0 from the controls, which both explained that their numbers are only in charts);
  - 512 of 512 fields right;
  - recall 32/34 (94%);
  - 0 of 6 QA-flagged rows were wrong (the flags were table rows that pypdf splits across lines).
- **Caveats:** small n (4 papers, one run, one model), and the verification was answered per table group.
- **In short:** "The LLM drafts the norms table; nothing it extracts is used until a human has checked it against the page. To measure hallucination instead of hoping, I also gave it two papers whose numbers exist only in charts, where any row would be invented. It returned none, all 32 real rows survived verification, and it missed 2 values buried in prose."

### DEC-027: The comparison engine: verified rows only, conventions converted, no invented unit conversions, a fixed ranking and a ±1 SD band (Phase 5)
- **Decision** (`kickcoach/compare/engine.py`):
  - **Inputs:** only `verified_norms()`, mapped feature → norm variable.
  - **Conventions and units:** a row is comparable only if its unit matches and its angle convention is known. Flexion angles become interior angles (180 − x; Xu et al. state 0° = straight on p. 3). Unclear conventions are refused.
  - **cm rows stay out:** the plant-foot distances (cm) are not compared with `plant_foot_offset` (leg lengths), because converting needs a leg length the video doesn't give.
  - **One row per feature,** by a fixed ranking: the player's level (a *heuristic* word list), adults over children, maximal over accuracy kicks, rows with a spread, larger n, then norm_id. The others are listed in `also_matching`.
  - **Verdict:** below / within / above the study's mean ± 1 SD, or its reported range. The band is *heuristic*, labelled so in the response and the UI.
  - **Caveats on every comparison:** lab method vs 2D video; for velocities, Nunome & Ikegami 2006 (2D distorts knee angular velocity in the last 50 ms); accuracy kicks are slower (N-011 vs N-012: 1364 vs 1175 deg/s); small n.
- **Corpus result** (`uv run python tools/run_assess.py` → `results/assess_summary.csv`):
  - **Knee at contact:** below Scurr & Hall's self-selected band (146.1–166.3°) on 19 of 22 accepted clips. E2 shows the pipeline agrees with Aamir's 2D labels on the 19 labelled clips (2.7° MAE), so the gap is between 2D video angles of these kicks and that study (7 recreational men, penalty kicks, 3D projected onto the sagittal plane). Its cause is not established, and it is reported as a comparison, not a fault.
  - **Velocities:** c13 (1314°/s) and c19 (1698°/s) are within 860–1720°/s (N-001).
  - **Trunk lean and plant foot:** no comparable norm.
- **In short:** "Every verdict comes from a human-verified row, converted to my angle convention by a rule I can point at. Where a paper used centimetres and I measure leg lengths, the app says 'no comparable study' instead of inventing a conversion."

### DEC-028: Four fault rules, each cited or heuristic; a threshold can come from a verified norm (Phase 5)
- **Rules** (`knowledge/faults.yaml`, `kickcoach/compare/faults.py`):
  - **F-LEAN-BACK:** trunk lean < −10°. *Heuristic:* about 4× E2's trunk MAE of 2.7°, so measurement error alone rarely fires it.
  - **F-SHALLOW-BACKSWING:** max knee flexion angle > 89.61°, i.e. N-032 (Xu et al. 2025: 102.71 ± 6.16° of flexion = 77.29° interior) + 2 SD. The 2 is *heuristic*. If N-032 weren't verified, the rule would be inactive and say why.
  - **F-PLANT-FAR-BEHIND:** plant foot offset < −0.25 leg lengths. *Heuristic*, uncalibrated, and this feature's accuracy is unmeasured (E2 doesn't score it).
  - **F-REVERSED-SEQUENCE:** shank peaks before thigh. Cited: Kellis & Katis 2007, p. 155 (proximal-to-distal sequence).
- **Templates:** no numbers except `{value}` (the magnitude; the words give the direction). Claims are cited (`text_source`) or marked heuristic coaching advice. A test enforces both.
- **On the 22 accepted clips:** lean-back 10, shallow backswing 5, plant foot 3, reversed sequence 0 (15 clips fire at least one).
  - **c24,** a coach's demonstration of "leaning back too much", fires lean-back (−12.6°). That's a sanity check, not calibration.
  - **c20 (Ronaldo)** fires shallow backswing (93.7°). The norm is from elite players measured with IMUs, and his knuckleball technique may differ. No rule is validated against kick outcomes.
- **In short:** "Every threshold is either cited or labelled heuristic with its reason, in the code and on screen. The backswing threshold is derived from a verified norm row, so if that row were rejected, the rule would switch itself off rather than keep a number nobody can source."

### DEC-029: Feedback is LLM-worded under a deterministic validator, with one retry and a template fallback (Phase 5)
- **Decision** (`kickcoach/feedback/`):
  - **Facts in:** the detected faults, measured and not-measured features (each with its definition), matched norm rows and the citable IDs. The model is `gemini-3.5-flash-lite` (the dev model, its own free quota).
  - **Structured output:** a hand-written JSON schema (no `$defs`/`$ref`), parsed with Pydantic.
  - **Validator:** schema-valid; exactly one item per detected fault; citations only from the given set; every number within a per-unit tolerance of a provided value (citation years exact; *heuristic* tolerances); no mention of a `not_measured` feature (*heuristic* phrase lists).
  - **On failure:** one retry with the errors, then the template. Both attempts are logged and stored.
  - **No call at all** when no fault fires or the clip is refused; a missing key or quota gives the template.
- **Evidence:**
  - **Corpus run** (`tools/run_assess.py --llm`): 15 of 15 clips with faults passed on the first try (responses in `results/feedback_llm/`). So on real outputs the validator never had to reject anything; its rejection paths are proven by unit tests with bad answers.
  - **The templates pass the same validator.**
- **Prompt v2:** each feature now comes with its definition, after v1 called a 98° interior angle "98 degrees of knee flexion" (82° in flexion terms). v2 still writes "max knee flexion … 98 deg", because the feature's own ID, `max_knee_flexion_backswing` (from the MVP spec), says flexion. The template words it correctly.
- **Limits:** the validator checks numbers, IDs, coverage and topics, not meaning. Causal claims like "leaning back … loses power" pass uncited. E7 (Phase 8: Aamir's ratings plus a judge) is where meaning gets scored.
- **In short:** "The LLM writes the words, never the numbers. A deterministic validator rejects any number that isn't a measured or provided value, any citation it wasn't given, and any mention of something we didn't measure. The output gets one retry with the errors, then a template, so the app works with no key at all."

### DEC-030: SQLite sessions with self-history, and a FastAPI app serving the adapted v1 page (Phase 5)
- **Storage** (`kickcoach/storage/db.py`):
  - **Tables:** D16's sessions, features, faults, feedback and artifacts, with stdlib `sqlite3` and one connection per call. The file is `data/kickcoach.db` (gitignored).
  - **History delta:** value − the mean of the same player's last 3 sessions in which the feature was measured. The 3 is *heuristic*, from the MVP spec's example.
- **API** (`api/app.py`, `uv run uvicorn api.app:app`):
  - **Endpoints:** `POST /analyse` (multipart: video plus player, level, kicking_leg, capture_fps, slow_motion, use_llm), `GET /sessions`, `GET /sessions/{id}`, `GET /health` and `/` (the page).
  - **Limits:** uploads ≤ 100 MB and ≤ 600 frames (*heuristic*).
  - **Data kept:** the video and its frames are deleted after analysis. The pose array, ball candidates and response are kept in `data/sessions/<id>/` (gitignored).
  - **Testability:** `create_app()` takes the DB path, session directory and analyser, so tests run without video or models.
- **Page** (`api/static/index.html`): v1's look is kept. v1's rating, pro reference, profiles and drills are removed. Every server string is escaped (v1 inserted LLM text unescaped), heuristic thresholds and bands are labelled, and `/?session=<id>` reopens a stored session.
- **Evidence:**
  - **Real uploads through uvicorn:** c07 took 88 s, was accepted, found contact at frame 76 (= label), fired 3 faults and got LLM feedback that passed on the first try. c10 then showed history deltas against session 1.
  - **Re-encoding shifts values slightly:** the uploaded files are the trimmed MP4s, an H.264 re-encode of the frames the cached pipeline ran on, so values move a little (c07 trunk −11.84 vs −11.60°; c10 knee 123.16 vs 122.39°).
  - **Not checked:** the page's rendering. Headless Brave failed in this session (`CVDisplayLinkCreateWithCGDisplay failed`, even on a `data:` page).
- **In short:** "The API is a thin layer. The measurement is a pure function of poses and metadata, and everything after it is deterministic except the wording, so I could test the whole response shape, including the no-API-key path, without decoding a single video."

### DEC-031: The numbers derived from the clips are committed, so every evaluation reruns from a fresh clone (Phase 6)
- **Decision:** commit the pose keypoints (`data/cache/pose/<model>/`, 4 models), the ball detections (`data/cache/ball/`) and each clip's `data/frames/<clip>/meta.json` (frame times, fps, image size), about 5.5 MB in all. Frames and video stay gitignored (CLAUDE.md rule 4); `.gitignore` re-includes only these paths.
- **E4's cache lookup** (`eval/e4_gemini.py: cached_record`): the cache key hashes the video's bytes, and a clone has no video. Without the file, the record is found by its tag (`e4 <clip> e4-v2 r<n>`) and must match the request in every field except the video hash; with the file, the exact lookup is used as before. Both lookups found the same 27 records.
- **Alternatives:**
  - document only the full rebuild (download the clips, ingest, run pose and ball): hours, needs YouTube, node and three sets of weights, and a source video can disappear;
  - commit only the CSVs: then the results can be read, not reproduced.
- **Why it's allowed:** the keypoints are coordinates, the same kind of data as the committed labels, and hold no pixels. The rule exists for the video's copyright and the players' privacy.
- **Evidence:** a copy of exactly the files git would track (no `.env`, no `data/raw`, no frames) ran `uv sync` and `uv run python -m eval.run_all`, and reproduced every file in `results/` byte for byte, including `summary.md`. The main repo's own rerun matched the Phase 5 results byte for byte too.
- **In short:** "Reproducibility meant shipping the intermediate numbers, not the video. Pose keypoints are just coordinates, like my labels, so I committed them, and the LLM caches are keyed so they can be found without the video. A clean copy with no API key regenerates every results file byte for byte in about half a minute."

### DEC-032: `eval/run_all.py` regenerates everything with the API key blanked, and writes the README's results table (Phase 6)
- **Decision:**
  - **Steps, in order:** the pipeline on every clip (from the caches), the cutoff tuning, E1, E2, E3, E4 (scoring), E5, E8, the repeatability floor, and `tools/run_assess.py --llm`.
  - **No API calls:** each step runs with `GEMINI_API_KEY=""`. python-dotenv doesn't override a variable that is already set, so `.env` is never read and a cache miss raises `NoApiKey` (checked with the installed dotenv). For feedback that means a visible template fallback, which the summary counts as a cache miss.
  - **Outputs:** `results/summary.md` (dataset, headline, the floor, a table per evaluation, the worst cases), built from the regenerated CSVs with each eval's own metric functions (E2's paired bootstrap, E3's `metrics`, E8's `confusion`). The headline table is also written into `README.md` between `<!-- results:start -->` and `<!-- results:end -->`.
  - **Not rerun:** the failure renders (they need frames) and the norm extraction (it needs the papers and a key; E5 scores its committed output).
- **Why the README block:** CLAUDE.md rule 2 says README numbers come only from scripts that ran. A generated block can't drift from the results, and when the blind relabel lands, one command updates the floor everywhere.
- **Evidence:** 34 s for every step on the M2. The summary's numbers match each `e*_summary.md`.
- **In short:** "One command rebuilds every table from committed inputs and rewrites the README's results block, so a number in the README can't drift from the code that produced it. It blanks the API key for the run, so it physically can't call Gemini; LLM results come only from the committed response cache."

### DEC-033: The labelling floor compares a blind relabel with the first labels, and quotes the points moved from the pre-fill (Phase 6)
- **Decision** (`eval/repeatability.py`): for each confirmed task in `data/labels/repeat/`, the point distances to the first label (pixels and thigh lengths, as E1) and the knee and trunk angles from each set of points (E2's geometry). It writes `results/repeatability_points.csv`, `_angles.csv` and `_summary.md`, and with no confirmed task it says "not measured yet" and writes no numbers.
- **Why the moved-only rows:** `--repeat` pre-fills from RTMPose, exactly as the first labelling did (DEC-013). A point left at the pre-fill both times agrees to 0 px by construction, which would make the labeller look perfectly consistent. The floor to quote uses only points, and angles with a point, moved in at least one session. Both are reported.
- **Status:** the blind relabel hadn't been done by Phase 6, so the README and summary say the floor isn't measured.
- **In short:** "Before claiming a model is accurate, I check how consistent my own labels are, by relabelling frames blind at least two days later. Because the tool pre-fills from a model, points I didn't touch agree trivially, so the honest floor only counts points I actually moved."

### DEC-034: The hosted demo runs in public mode: nothing stored on the server, history kept in the visitor's browser, template feedback only (Phase 7)
- **Decision** (`KICKCOACH_PUBLIC=1`, `api/app.py`):
  - **No server-side storage:** no SQLite row, pose array or response file; `/sessions` returns 404. The upload and its frames are deleted when the job ends, as in local mode.
  - **History in the browser:** the page keeps each result in `localStorage` (at most 20, *heuristic*) and sends the earlier sessions' measured values with an upload (`history` form field). The server computes the delta with the same function the database path uses (`db.delta_vs`, last 3 measured sessions, DEC-030).
  - **Feedback is always the template:** `use_llm` is forced off, whatever key is set (a test sets one and checks it isn't used), and the page says why.
  - **Local mode is unchanged** (the default): sessions in `data/kickcoach.db`, LLM feedback when a key is set.
- **Why, from the current Gemini API terms** (effective 23 March 2026, read this phase):
  - unpaid services: "Google uses the content you submit to the Services and any generated responses to provide, improve, and develop" its products, and "human reviewers may read, annotate, and process your API input and output";
  - "Do not submit sensitive, confidential, or personal information to the Unpaid Services";
  - "You may use only Paid Services when making API Clients available to users in the European Economic Area", and likewise Switzerland and the UK. A public URL can't exclude those users, so the free tier can't serve the demo at all; a notice wouldn't be enough.
- **Alternatives:** a paid Gemini tier (spending: an ask-before, and not needed to show the feature: the repo's evaluated LLM feedback already does); a notice plus the free tier (ruled out by the regional clause); keeping strangers' sessions on the server (no need for the demo, and less to hold is less to leak).
- **In short:** "The demo can't use the LLM I evaluated, because the free tier's terms forbid serving EEA and UK users and may use inputs for training. So public mode switches to the deterministic templates and keeps nothing server-side: history lives in the visitor's browser and is sent back only to compute the comparison."

### DEC-035: The demo is hosted on Modal's free credits: one 4-core container, analyses run as polled jobs (Phase 7)
- **Host** (`deploy/modal_app.py`, chosen from the options below):
  - **Hugging Face Spaces:** Docker and Gradio Spaces now need PRO ($9/month) to create (docs, read this phase); free accounts get static Spaces and 2 ZeroGPU Gradio Spaces only. CPU Basic is 2 vCPU / 16 GB.
  - **Render and Koyeb free tiers:** 0.1 CPU and 512 MB, far too little for RTMPose plus two YOLOX-m passes.
  - **Modal Starter:** $30/month of free compute. Sign-up needed no card. At $0.0000131 per core-second, a 100 s analysis on 4 cores costs about half a cent.
  - **The cost of 2 vCPU,** measured on the M2 with onnxruntime capped: one 150-frame clip takes 96 s with all cores, 160 s with 2 threads and 267 s with 1, so a 2-vCPU host would take roughly 3–6 minutes a kick.
- **Shape:**
  - the same FastAPI app as local, with `KICKCOACH_PUBLIC=1`;
  - `cpu=4.0` (physical cores), 4 GB, `max_containers=1`, scale-down after 300 s idle (*heuristic*);
  - weights fetched at build: YOLOX-m checked against its sha256, RTMPose via rtmlib.
- **Jobs:** Modal caps any HTTP request at 150 s, and an analysis takes one to two minutes. So `POST /analyse` returns 202 with a job id, one worker thread runs jobs in order, and `GET /jobs/<id>` reports the stage, frame counts and queue position, then the result.
  - Job ids are random (`secrets.token_urlsafe(12)`), so one visitor can't read another's result. Results are held for 30 minutes (*heuristic*).
  - At most 4 jobs wait (*heuristic*); beyond that the server returns 503.
  - One container keeps the queue in memory, so every poll reaches the job.
- **Thread cap** (`kickcoach/pose/ort_threads.py`, `KICKCOACH_ORT_THREADS=4`):
  - rtmlib builds its onnxruntime sessions without options, and the 4-core container reports `os.cpu_count()` = 20.
  - The cap matches the quota, but it made **no measurable difference**: c07 took 60–107 s capped (6 runs, median 70 s) and 60–128 s uncapped (3 runs, median 73 s). Host variance dominates. My first single-run comparison suggested 1.8× (JOURNAL Phase 7).
  - It stays as a guard and a benchmarking knob. It changes outputs only by summation order: ≤ 0.0004 px on keypoints and 3e-6 on ball boxes over 20 frames of c07.
- **Evidence, against the deployed URL** (`deploy/smoke_test.py`):
  - c07 accepted, contact at frame 76 (= label), the same 3 faults as local, template feedback, no session id;
  - a portrait-phone version of c07 came out upright, with the same contact and faults;
  - c26 refused with the gate's reason;
  - a 34 s source video was rejected: "trim it to the one kick (at most 600 frames)".
  - Values are within 0.2° of the local run (different CPU, float order).
- **In short:** "Picking the host was a compute question, so I benchmarked the pipeline at 1, 2 and all threads before choosing. Hosts cap a request at 150 s and an analysis takes longer, so it became a polled job queue with real progress. I also caught myself over-reading one benchmark: a single run said a thread cap made it 1.8× faster, but repeats on the live host showed the spread between runs was bigger than the effect."

### DEC-036: The page rebuilt: it starts empty, the example is opt-in, the stick figure comes from the measured keypoints, and phone rotation is applied at ingest (Phase 7)
- **Review of the Phase 5 page:** it looked generic and template-made, and it opened with statistics already showing. That was the stored session 1 at `?session=1`, the URL the Phase 6 handoff pointed to, but on a fresh app it reads as fake data.
- **Page** (`api/static/index.html`, plain HTML/CSS/JS, no external fonts or libraries):
  - **Look:** system fonts, a neutral light theme with dark mode, one accent colour; v1's neon-on-black display style is gone.
  - **Starts empty:** an upload panel with filming tips. Results appear only after an upload, a saved session is clicked, or the example is asked for.
  - **Real progress:** upload percentage, then the job's stage and frame count, plus elapsed time.
  - **Results:** the clip verdict, a stick figure, a measurements table with the study band and source, then faults with their feedback and threshold source. Clip checks and the raw JSON sit behind a disclosure. Internal decision ids (e.g. "(DEC-010)") are stripped from the reasons shown.
  - **History:** listed from the browser's storage only.
- **Stick figure** (`kickcoach/figure.py`): the cleaned, filtered keypoints at contact and at the deepest backswing, which are what the angles were measured on. It is drawn as SVG with the kicking leg, the knee arc and the trunk-vs-vertical line marked. There is none for a refused clip, since drawing one would suggest a measurement the gate declined. It holds no pixels.
- **Example** (`tools/make_example.py` → `api/static/example.json`, also run by `eval/run_all.py`): c07 from the committed caches. The script first checks the pipeline reproduces `results/pipeline/c07.json` exactly, and the feedback is the cached LLM answer that passed the validator. Opt-in, labelled as an example, and linked to its source video (Aamir approved the opt-in example).
- **Rotation** (`kickcoach/ingest/video.py`): phones store portrait video as landscape pixels plus a display rotation. Ingest now turns each frame upright by PyAV's `frame.rotation` (counterclockwise, like `np.rot90`), and `meta.json` gains `frame_size`. Before this, a portrait upload would have reached the pose model on its side. The committed clips have no rotation, so no result changed (`run_all` byte-identical).
- **Checked visually this time:** headless Brave still fails, so a 60-line Swift `WKWebView` snapshot tool (scratchpad, no download) rendered the page at 1280 px light and dark and at 390 px, against both local and deployed servers. It found an overlapping knee label, a missing space in author names and a table overflowing on phones, all fixed.
- **In short:** "The only picture the app keeps is a stick figure drawn from the exact filtered keypoints the angles came from, so the visual can't disagree with the numbers, and it holds no pixels of the player. The page also had to look like it starts fresh, so the example result is opt-in and clearly labelled."

### DEC-037: The coaching corpus: 55 verbatim chunks from 10 approved sources, open-licensed in full, copyrighted ones as short excerpts (Phase 8)
- **Approval:** "A + B + C" was picked from a list of 13 candidates (2026-11-02):
  - A: the 4 norm papers already on disk;
  - B: openly licensed new sources (Shan & Zhang 2011 review, a 2024 upper-body review, Wikipedia's "Shooting", wikiHow "Kick a Ball", a PLoS ONE attentional-focus study);
  - C: free-to-read but copyrighted coaching material (Special Olympics' football guide and two state sheets, The FA's finishing article), as excerpts of at most 60 words with the URL.
- **What survived the download:**
  - the 2024 review (Springer) and wikiHow answered a script with a CDN bot-challenge page. The auto-mode classifier stopped my attempt with a browser user-agent as bot-check circumvention, rightly; both are dropped;
  - SO Wisconsin's "Shooting Faults and Fixes" chart is from their *basketball* guide; dropped;
  - 10 sources remain (`tools/coaching_sources.py`, `knowledge/coaching_corpus.yaml`).
- **Chunks:** 55, each a verbatim excerpt with source, page/section and URL. `tests/test_retrieval.py` checks every chunk against the downloaded text when it is present: letters and digits only, so PDF hyphenation can't fake or hide a mismatch. 2 chunks failed the first check (one spanned a page break, one was over 60 words) and were fixed. The source files stay out of git (`knowledge/coaching_sources/`).
- **Tags** (handoff D14): `faults` (which rule's aspect of technique a passage addresses), `kick_type` (instep / general / other), `level`. The *kind* of help (why / cue / drill) is deliberately not a tag, so E6 can test whether embeddings find it.
- **The corpus has almost no drills:** only the plant foot has one (SONC's "stand with non-kicking foot next to the ball, phantom kick"). Drills for the other faults stay uncited.
- **In short:** "Every passage the LLM can cite is a verbatim quote that a test checks against the source, with the URL kept. Where the sources had no drill for a fault, I reported a corpus gap instead of writing one and calling it retrieved."

### DEC-038: Retrieval is a fault filter then gemini-embedding-2 similarity; E6's relevance labels are Claude's, frozen before any embedding (Phase 8)
- **Embeddings:** `gemini-embedding-2` at 768 dimensions:
  - it's free on the free tier, and the docs' recommended model (2026-11-03);
  - the corpus is public text, so the free tier's "used to improve our products" doesn't matter here;
  - it takes no `task_type`: documents are sent as `title: none | text: …` and queries as `task: search result | query: …`. `title: none` keeps a chunk's location (e.g. a "drill" column) out of what E6 tests;
  - ⚠️ **a plain list of strings returns one aggregated embedding** with this model, so each text goes in as its own `Content`, and `embed_texts` raises if the count is off;
  - vectors are committed (`knowledge/embeddings/gemini-embedding-2_768.npz`, 0.2 MB).
- **Queries** are a fixed function of (fault rule, need): "Leaning back at contact. Why does this matter for the kick?" and so on. All 12 are embedded in the committed cache (a test checks), so retrieval never needs a key, including on the public demo.
- **Filter:** chunks tagged with the fault, kick type instep or general.
- **Relevance labels** (`eval/e6_relevance.yaml`):
  - written by Claude before any embedding existed, with stated criteria, a strict reading of "about this aspect", why/cue/drill defined;
  - sha256 `274edf71…` at 13:08 IST on 3 Nov, recorded before the first embedding call;
  - not human ground truth in DEC-007's sense, but not independent either: the same person wrote the tags. The README says so.
- **E6 result** (`results/e6_summary.md`), 9 queries with a relevant chunk (3 drill queries have none):

  | method | Recall@3 | MRR |
  |---|---|---|
  | filter only (expected over a random order) | 0.46 | 0.61 |
  | filter + embeddings | 0.56 | 0.75 |
  | embeddings only | 0.29 | 0.50 |

  By MRR, filter + embeddings beats the filter on 6 of 9 queries and loses on 3. The filter does most of the work (embeddings alone are clearly worse). The embeddings add a modest edge that 9 queries can't pin down.
- **Decision:** the app uses filter + embeddings (the best of the three here), with that small-n caveat.
- **In short:** "With 55 chunks, the metadata filter did most of the work: embeddings alone were worse than the filter, and filter-plus-embeddings beat the filter on 6 of 9 queries. I labelled relevance before computing any embedding and recorded the file's hash, because I also wrote the tags."

### DEC-039: The LLM gets retrieved passages (prompt v3); v2 stays byte-identical, and templates never use retrieval (Phase 8)
- **Per fired fault:** the top 2 passages for each need (why, cue, drill), deduplicated, typically 4–6 (*heuristic*). They go into the facts as `coaching_passages`, and their IDs join `citable_sources`.
- **Prompt v3** adds one rule: use a passage only where it really supports what you write, in your own words; cite its ID; copy no numbers or names from it; write uncited where nothing fits. Without passages the prompt is v2 word for word, and all 15 cached v2 answers still hit (checked).
- **Validator:** unchanged. A citation must be one of the IDs given, so a passage the search didn't return can't be cited.
- **Output:** the feedback block gains `passages` (IDs per fault) and `sources` (citation, page, URL). The page renders a cited passage as a link to its source.
- **Default:** `assess(retrieval=True)` for LLM feedback. Templates (and so the public demo) never call retrieval; the demo's opt-in example now shows the cached v3 answer, with its source links.
- **The first real validator rejection** (v3, c07, 3 Nov): the model cited `Kilic_2021`, a source it was never given. The retry, with the error, passed.
- **In short:** "Retrieval widened what the model may cite, not what it may claim: a citation has to be one of the passage IDs the search returned, and the first real rejection was exactly that, a made-up 'Kilic 2021'."

### DEC-040: E7's protocol, fixed before anyone rated: three variants, a 20-item blind sample for a human rater, an LLM judge with the same view, and a κ ≥ 0.6 bar (Phase 8)
- **Items:** every fault item on the corpus (18: 10 lean-back, 5 shallow backswing, 3 plant foot; the shank-before-thigh rule never fires) in three variants: LLM + passages (v3), LLM alone (v2), template. 54 items (`results/e7/items.json`).
- **The human sample** (seed 20261103, then frozen so rebuilding the items can't change it):
  - 8 (clip, fault) pairs in both LLM variants, stratified 3 / 3 / 2 by fault;
  - plus 4 templates (2 / 1 / 1);
  - shuffled. The page never receives the variant. Blinding is partial: a template's wording repeats.
- **Rubric:** 1–5 on correctness (true to the facts and the cited sources; an unsupported claim counts against it), specificity (about this kick) and actionability (could the player do it next session, alone). Anchors at 1, 3 and 5 are shown beside the buttons.
- **The page** (`tools/rate_feedback.py`) shows the facts the writer was given and the text of every cited source, so correctness can be checked rather than guessed.
- **Judge:**
  - planned: `gemini-3.6-flash`, a different model from the generator (`gemini-3.5-flash-lite`), so it isn't grading its own wording;
  - **changed to `gemma-4-31b-it` before any agreement was computed.** `gemini-3.6-flash`'s free tier allows 20 requests a day (the 429's QuotaFailure on 3 Nov), and it ran out after 21 of 54 items, only 9 of them Aamir's. Agreement needs all of his 20 under one judge, and the free Flash models each have the same 20-a-day ceiling. Gemma 4 is free on the Gemini API, is from another model family than the generator, and has open weights, so the judge could be rerun locally;
  - Gemma ignores the JSON response type enough to add a stray code fence after the object, so the scores are parsed from the first `{…}` block and checked to be integers 1–5;
  - the 21 `gemini-3.6-flash` answers are kept, and the report compares the two judges where both rated;
  - the same view and rubric for both; temperature 0; cached in `results/e7/judge_raw/<model>/`.
- **Bar:** the judge stands in for the human rater on a criterion only if the quadratic weighted κ on the rater's 20 items is ≥ 0.6 (*heuristic*: Landis & Koch's "substantial" band starts at 0.61). Spearman, exact and within-1 agreement are reported too.
- **Status at the end of Phase 8:**
  - no human ratings existed yet, so agreement and the variant comparison were pending.
  - Gemma judged 30 of 54 items (hangs and 503s through the evening, 8 resumable rounds).
  - The two judges agreed on correctness (κ 0.64, 21 items) but both sit near the ceiling: Gemma gave specificity 5 to every item.
  - A warning sign for the judge, not a result: on c07, Gemma gave correctness 5 to "Leaning back causes your shots to rise too high and lose power", whose cited passage supports "rise" but not "power"; Flash gave 4.
  - The bar is what decides whether either judge is used.
- **Judge again `gemini-3.6-flash` (7 Nov):** Gemma stalled at 30 of 54. Flash's next free day covered the rest of the 20-item sample (`judge --model gemini-3.6-flash --sample-only`), so Flash is the judge and Gemma the second. No human agreement existed at either switch, so neither was chosen for its scores. On the 20, Flash gave correctness 5 to all 20 (see DEC-041).
- **In short:** "I fixed the agreement bar before anyone rated, so the LLM judge had to earn its place against a human. The first look was a warning: the judge gave full marks for correctness to a sentence whose cited source didn't support half of it."

### DEC-041: E7 runs without a human rater; LLM ratings are reported as LLM ratings, never as human ratings (Phase 8, 7 Nov)
- **What happened:** no human rating was made. Ratings from an external AI tool that simulated five raters (players aged 13–30 and a youth coach) were available instead, and Claude rated critically too.
- **Decision:**
  - neither set goes into `results/e7/aamir_ratings.json`;
  - the panel's scores are stored verbatim in `results/e7/raters/persona_panel.csv`, labelled AI-generated (the tool's model wasn't recorded);
  - Claude's ratings are in `results/e7/raters/claude.json`: written rules, a reason per item, and `"blind": false`, because Claude had seen the variant mapping in this chat;
  - E7 reports them as LLM raters and says no human rated.
- **Why:** E7's claim was "an LLM judge, checked against a human". With AI scores in the human slot, the judge's agreement would be AI agreeing with AI, while the README would say "human". CLAUDE.md rules 1–2 exist for exactly this.
- **Consequences:**
  - the κ ≥ 0.6 bar can't be applied, so the judge isn't validated and isn't used beyond the sample;
  - no claim may say "validated against human ratings";
  - what E7 *can* say is limited to what the LLM raters show, with their caveats.
- **Fact-check of the panel** against `results/e7/items.json`:
  - its quoted values (−12.65, 100.89, 89.61, 93.67, −0.36, +5.36, the simultaneous peaks) are right;
  - R11's contradiction is real;
  - its "R01/R06 duplicates" is wrong (R01 is −12.65, R06 is −17.35; the pairs are R01/R09 and R03/R06);
  - some personas score their own goals (hip timing, weaker-foot passing, laces contact) rather than the rubric. One tool wrote all five, so they aren't independent.
- **In short:** "With no human rating available, I didn't let synthetic ratings stand in for one. They're in the evaluation under their real name, and the claim shrank to what LLM raters can support."

### DEC-042: E7's three slips become validator checks; prompt v4 asks for the value in "what"; E7's items are frozen (Phase 9)
- **What E7 found** (DEC-041, by reading the rated items): a knee interior angle written as an amount of bend ("your knee bend … was 101 degrees" for a 101° angle, i.e. 79° of bend: R02, R13), a direction the kick didn't have (R11: "makes your body lean away" on a kick that leaned 5.36° forward), and v3 dropping the measured value from "what" in 7 of 8 pairs.
- **Validator** (`kickcoach/feedback/validate.py`), two new checks, both phrase tables marked *heuristic*:
  - **6, direction claims:** a phrase claiming a sign ("lean back/away", "behind the ball"; "lean forward", "ahead of the ball") is rejected when that feature's measured value has the other sign. The fault's own direction is checked in "what" and "why_it_matters"; the corrective direction (the normal wording of advice) only in "what".
  - **7, knee angle as bend:** a sentence that restates a knee interior angle (measured, threshold or norm) with a bend word and no "angle" is rejected.
  - Neither reads meaning: a negated claim is rejected too, which costs a retry.
- **Checked before any new call:** on every cached Phase 8 answer (v2 and v3 first attempts), the checks reject exactly 5: R02, R11, R12 and R13 from the rated sample (R12 is the same knee slip, softened by "which is too straight"), and the unsampled c03 v3 ("bent to 90 degrees … too little bend"). A scan of all 31 cached answers found no rejection of a correct sentence.
- **Prompt v4** (the path with passages; `feedback-v4`): v3 plus three rules: put the fault's measured value and its direction in "what"; call a knee number an angle, never bend or flexion; describe the kick only as its measurements show. The path without passages stays v2 word for word.
- **Rerun** (`tools/run_assess.py --llm`, 16 flash-lite calls):
  - 14 of 15 clips passed on the first try. c24's first answer said "Your maximum knee flexion in the backswing was measured at 100.89 deg", which check 7 caught; the retry wrote "… resulted in a knee angle of 100.89 deg" and passed. So the new check fired on live output in its first run.
  - "What" now carries the measured value in **18 of 18** fault items (the frozen v3 texts of the same items: 3 of 18).
  - Cost: stiffer wording ("which points to leaning back"; "-0.36 leg_lengths" once).
- **E7's items are frozen** (`eval/e7_feedback.py prepare` is a no-op unless `--rebuild`): every rating refers to a v2/v3 text, so rebuilding with v4 would put new texts under old ratings. E7's summary gains a computed section, "Today's validator on the rated texts": it rejects 5 of the 36 rated LLM texts.
- **In short:** "The human-style review found three kinds of slip my validator passed. I turned each into a deterministic check and replayed it on the frozen, already-rated texts: it rejects exactly the items the raters marked down, and in its first live run it caught a new instance and forced a retry."

### DEC-043: The coach chat: four read-only tools that compute every number, manual function calling, an answer checker, and E9 (Phase 9)
- **Where it's justified** (DEC-008): an open question ("why did my knee angle drop?") needs different data each time, so the model chooses the tools. The analysis pipeline stays linear (D15). Local mode only: `/chat` returns 403 in public mode and the page hides the panel (DEC-034).
- **Tools** (`kickcoach/chat/tools.py`), all read-only, with the player fixed when the chat starts (the model can't read another player's sessions):
  - `get_history(feature_id?, last_n?)`: sessions newest first; with a feature, its trend, with the change and step changes already computed, E2's mean absolute error for that feature (read from `results/e2.csv`, never copied), whether the change exceeds it (*heuristic*: one MAE) and a sentence saying so;
  - `get_session_features(session_id?)`: values, units, definitions, not-measured reasons, faults with thresholds and their sources, or why the clip was refused;
  - `compare_to_norms(feature_id, session_id?)`: the comparison engine's verdict for that feature plus the fault rule on it;
  - `search_drills(query, fault_id?)`: the coaching corpus, with the fault filter when a fault is named (E6's best row), or embeddings alone. Free-text query vectors go to their own cache (`results/e9/query_embeddings.npz` for the eval; the app's is gitignored), so the committed corpus file doesn't grow with every question.
  - Faults and comparisons are recomputed with today's rules on the stored measurements.
- **Loop** (`kickcoach/chat/agent.py`, prompt `chat-v1`): `gemini-3.5-flash-lite`, the SDK's automatic function calling off, so every call is logged and checked. At most 4 tool rounds (*heuristic*), then the mode `NONE` forces an answer. The model's turns go back unchanged, because Gemini 3 attaches thought signatures to function-call parts; the cache key leaves the signatures out, so a conversation with deterministic tool results replays from the cache with no key (checked on the real API on the first question).
- **Checker** (`kickcoach/chat/validate.py`), reusing the feedback validator's parts:
  - every number must be one a tool returned (or one from the question), within the per-unit tolerance;
  - a number written with a unit must match a value in that unit;
  - every K-, N- or F- ID must have been returned by a tool;
  - a sentence about a feature no tool ever returned as measured must say it wasn't measured;
  - no knee angle called bend.
  - One retry with the errors, then a deterministic answer listing what the tools returned. With no key, a template lists the latest session.
- **E9** (`eval/e9_chat.py`, `eval/e9_questions.yaml`): 15 scripted questions fixed before any run (sha256 `f2ca9528206ff82e`): facts, trends, norms, drills, two not-measured refusals, a refused clip, an off-topic question and a multi-tool one. They're asked of two **fixture players**: real pipeline outputs from different people's clips, stored as sessions with made-up dates. Expected values are references to clips, so the scorer reads the numbers instead of a typed copy. The metrics:
  - tool choice and arguments;
  - numbers in first drafts and final answers that a tool returned;
  - expected values restated;
  - refusals;
  - heuristic wording checks.
- **Three runs, all kept in the record:**
  - **Run 1** (`results/e9/run1/`): right tools and arguments on 15 of 15. First drafts passed the checker on 12 of 15, but 2 of the 3 rejections were the checker's fault: history values and fault thresholds came without units, so correct roundings ("90" for 89.61) got the strict default tolerance. Worse, a real error passed: "plant foot offset went from -0.02 m to -0.32 m" (leg lengths). The model wrote a unit the tool never gave it.
  - **Fix:** units in `get_history` and in fault entries; the checker takes units from feature IDs and rejects a number written in the wrong unit. Replaying run 1's drafts through the new checker accepts the roundings and rejects "-0.32 m".
  - **Run 2:** 15 of 15 again; first drafts passed on 13 of 15. q03's rejection was right this time: the model had done arithmetic ("changed by 17°" = 98 − 81), which rule 2 forbids. Two failures passed every check:
    - q03's draft gave the missing velocity an invented reason ("insufficient high-speed markers");
    - asked on the page, "Is my backswing getting deeper or shallower?" got "getting shallower" for a +4.37° change against a 4.63° error.
  - **Fix:** `get_history` now says why a value is missing (the real reason), and each trend carries a sentence comparing the change with the error. That question was added as **q16, marked post hoc**.
  - **Run 3 (final, `results/e9_summary.md`):** right tools with the expected arguments on 16 of 16; 55 of 55 numbers in first drafts came from a tool; first drafts passed the checker on 15 of 16 (q06 cited N-032 for "there is no norm", fixed on the retry); both refusals correct; q16 now says the change "can't be told apart from measurement noise"; 33 model calls.
- **Limits:**
  - one run each, of a non-deterministic model, and a run reuses any turn whose request didn't change, so the runs aren't independent samples;
  - Claude wrote the questions, the expectations and the agent;
  - the checker sees numbers, units and IDs, not meaning: an invented reason or an overclaimed trend passes it. Both were fixed by giving the model the fact it lacked, not by the checker;
  - the 1° tolerance lets "93" stand for 93.67 (run 3, q16);
  - jargon from stored reasons ("(DEC-010)") reaches the user.
- **Renumbering:** the chat's eval took E9, so SportsPose's 3D check (Phase 10) is now E10.
- **In short:** "Tool calling earned its place only in the chat, where the next lookup depends on the question. The tools do all the arithmetic, including whether a change is bigger than the measurement error I'd measured. A checker grounds every number and unit in what the tools returned. The best lesson came from the eval: every time a tool result left something out (a unit, a reason, whether a change was noise), the model filled the gap with something plausible and wrong. The fix each time was to put the fact in the tool's output."

### DEC-044: Phase order after the build: the write-up (Phase 10), then a system evaluation of its own (Phase 11), which takes in SportsPose's E10 (after Phase 9, 9 Nov)
- **What happened:** with the build done (Phases 0–9), the order was set: the write-up first, then evaluate the finished system as a stage of its own. The SportsPose access email was sent on 9 Nov; access can take weeks.
- **Decision:**
  - Phase 10 redrafts the project summary from the current results and aligns the README's opening;
  - Phase 11 tests the frozen system on unseen data: protocol hashed first, fresh phone kicks labelled before the pipeline sees them, end-to-end runs through the live app, repeatability, a small user test, and E10 if access is granted (`docs/PLAN.md` § Phase 11);
  - SportsPose is no longer a phase of its own. Its check is one part of Phase 11, so a slow reply blocks nothing.
- **Why a separate stage:** every eval so far ran on the 26 development clips. Even the 5 held-out clips were labelled and looked at while building, and none is a phone video, which is what users will upload. Only data gathered after the freeze can say how the product does in use.
- **Consequence:** the summary is written before Phase 11. If Phase 11 gives a number on unseen phone video, the summary is updated.
- **In short** (the plan; Phase 11 hasn't run): "Each component has its own eval, but all on clips I looked at while building. So the last stage freezes the system, writes the protocol down first, and tests it on fresh phone videos labelled before the pipeline sees them. The number that counts is the one from data the system has never seen."

### DEC-045: The public repo holds project facts only; private working files stay out of it (before going public, 14 Nov)
- **Decision:**
  - `docs/STATUS.md` (the session to-do list and machine notes) and `docs/RESUME_BULLETS.md` are gitignored and untracked; personal working notes for the coding assistant live in the gitignored `CLAUDE.local.md`, which `CLAUDE.md` imports;
  - reports, slides and photos in `docs/` are gitignored (`docs/img/` keeps the diagrams);
  - the public docs keep every decision and its evidence, and who labelled or verified what (the author or Claude), but not working-style remarks, personal machine details or third parties' contact details. The per-entry summary is called "In short".
- **Alternatives:** untrack JOURNAL, PLAN and the handoff too (they're linked from code comments and the README, so the links would break); keep everything tracked (the to-do list and machine notes aren't project facts).
- **Consequence:** the git history still holds the earlier text and the committer metadata, so going public needs a fresh history (one squashed commit), not just a new commit.
- **In short:** "The repo records how the system was built and measured: decisions, evidence and failures. Session notes and personal planning stay local."
