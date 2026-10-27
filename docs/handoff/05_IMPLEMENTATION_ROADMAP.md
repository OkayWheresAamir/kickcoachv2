# KickCoach v2 — Implementation Roadmap

Vertical slices in dependency order. Time boxes are targets. If a slice overruns, apply its **cut line** and move on. Each slice ends with tests run, real output shown, and a `docs/DECISIONS.md` entry.

---

## Phase 1: resume version (about one long day)

### S0. Audit and setup (0.5 h)
- Read the old `kickcoach-2` repo and its July 2026 plan documents. Write `docs/V1_AUDIT.md`: the pipeline as it actually works, the bugs found, and the conflicts with this handoff.
- Create the repo skeleton, `pyproject.toml` or `requirements.txt`, `.gitignore` (data/raw, data/frames, .env), pytest, and ruff.
- **Accept when:** `pytest` runs and the audit is written.
- **Learn:** how the v1 pipeline flows end to end, and why the 16,444°/s value happened (find it in the code if possible).

### S1. Ingest and early risk check (1.5 h)
- `tools/fetch_clips.py`: downloads the manifest clips **locally** (run on the author's machine). Trims to the start and end times.
- `kickcoach/ingest`: reads the stream metadata (ffprobe: `r_frame_rate`, `avg_frame_rate`, duration), extracts frames, and records `fps_trusted`.
- **Early risk check (20 min):** send 3 clips to Gemini with the E4 prompt and look at the answers. This is not the evaluation, just a gut check that the headline comparison is worth running. Report what came back.
- **Accept when:** frames and metadata are extracted for all manifest clips. Aamir has approved the manifest of 20–30 clips (minimum 15).
- **Cut line:** 15 clips.
- **Learn:** container frame rate vs capture frame rate; why slow-motion clips break velocity calculations.

### S2. Pose adapters (1 h)
- A canonical skeleton definition plus adapters for MoveNet (port from v1), an RTMPose variant (`rtmlib`; prefer one with foot keypoints), and one more model (MediaPipe, YOLO11-pose, or ViTPose; whichever installs cleanly).
- Cache pose outputs per clip and model as `.npy`.
- **Accept when:** all models run on all clips, and the outputs share the canonical shape.
- **Cut line:** two models (MoveNet + RTMPose).
- **Learn:** top-down vs bottom-up pose estimation; COCO-17 vs foot keypoints; what keypoint confidence does and doesn't mean.

### S3. Labelling tool and labelling session (2 h, mostly Aamir's time)
- `tools/label_tool.py`: a simple OpenCV or matplotlib click-and-drag tool. It pre-fills from the RTMPose output. Aamir marks the contact frame and max-backswing frame, corrects the keypoints, and confirms each frame. Saves `data/labels/<clip_id>.json`.
- Repeatability subset: 10 frames relabelled later, blind to the first labels.
- **Claude Code builds the tool. Aamir does all the labelling.**
- **Accept when:** every evaluation clip has confirmed labels, and the repeatability subset is done.
- **Cut line:** 2 frames per clip (contact + max backswing).

### S4. E1 + model choice (1 h)
- `eval/e1_pose.py`: normalised error, PCK, missing rate. Kicking-leg joints and contact frames reported separately.
- Aamir chooses the model from the evidence. Record it in `DECISIONS.md`.
- **Accept when:** `results/e1.csv` and the summary table exist, and the decision is recorded.
- **Learn:** PCK; why normalise by thigh length; why contact frames are hardest (motion blur).

### S5. Signal processing, events, features, quality gate (2.5 h)
- Gap fill → leg-swap fix → Butterworth `filtfilt` → central differences.
- Kicking-leg identification; ball detection (YOLO "sports ball") → contact frame, with the foot-speed fallback; backswing frame.
- The Phase 1 features from `02_MVP_SPEC.md`, sanity bounds (cited), `fps_trusted` gating.
- Quality gate: viewpoint score + keypoint confidence + ball visibility → accept, or refuse with reasons.
- Unit tests: the synthetic stick-figure kick, leg swaps, bounds, geometry.
- **Accept when:**
  - synthetic tests recover the known angles and contact frame within tolerance;
  - the pipeline runs on all clips;
  - no value beyond the sanity bounds appears in the output.
- **Cut line:** drop `ankle_angle_contact` and `plant_foot_offset` if the foot keypoints or ball detection aren't reliable. Record why.
- **Learn:** zero-lag filtering; cutoff frequency vs frame rate; why derivatives amplify noise.

### S6. E2, E3, E8 (1 h)
- Angle accuracy (v1-style vs v2), contact detection (ball vs foot speed), and reliability (bounds violations, gate precision and recall).
- **Accept when:** result tables are generated, plus a failure analysis of the 3–5 worst cases.

### S7. E4 frontier baseline (1 h)
- Run Gemini on all evaluation clips with the fixed prompt and schema. Cache the responses and score them against the same labels.
- **Accept when:** `results/e4.csv` exists and the comparison table is in the summary, whatever the outcome.

### S8. Norms table and fault catalogue (1.5 h)
- An LLM-assisted extraction script reads 2–4 open-access papers (start with Kellis & Katis 2007) and outputs candidate rows against the D12 schema. Aamir verifies each row. E5 is scored from his corrections.
- `knowledge/faults.yaml`: 3–5 faults (for example: leaning back at contact, insufficient backswing knee flexion, plant foot far behind the ball, reversed segment sequence). Each threshold is cited or marked `heuristic`.
- Comparison engine: deterministic. Only verified norms are used. Caveats are attached.
- **Accept when:** at least 5 verified norm rows exist, the faults fire correctly on unit-test inputs, and E5 has been reported.
- **Cut line:** 3 variables; faults marked `heuristic` where no source is found. Never invent a number.

### S9. Feedback, storage, API (1.5 h)
- Pydantic feedback schema → Gemini structured output → validator (D13) → one retry → template fallback.
- SQLite sessions plus self-history deltas.
- FastAPI endpoint returning the `02_MVP_SPEC.md` response. Reuse or minimally adapt the v1 frontend to show quality, features, faults, and feedback.
- **Accept when:** an end-to-end upload works, the validator tests pass, and the system runs with no API key.
- **Learn:** structured outputs; why validate LLM output deterministically.

### S10. README and results (1 h)
- `eval/run_all.py` regenerates `results/summary.md`.
- The README covers: problem, architecture diagram, capture protocol, evaluation results with n, the labelling repeatability floor, decisions (link to DECISIONS.md), limitations, and how to run.
- Draft the resume bullet from the **actual** numbers, using the template in `04_EVALUATION_PLAN.md`.
- **Accept when:** a fresh clone plus the documented steps reproduce the summary from the committed labels and cached outputs.

**Phase 1 total:** about 14 hours. If time runs short, protect this priority order: **S0–S7, then S10**. Then S9 with template-only feedback, then S8.

---

## Phase 2: after the resume submission
- **S11. RAG coaching corpus:**
  - curate 30–60 tagged chunks from open-access or permitted sources;
  - build the retriever;
  - run E6, comparing the metadata filter alone with filter + embeddings;
  - pass retrieved passages into feedback with citations;
  - run E7.
  - Add RAG to the resume only after E6 and E7 have run.
- **S12.** A self-history trends view in the user interface.
- **S13.** Deployment (backend + frontend), with an upload size limit and video deletion after processing.
- **S14.** More clips and the held-out set expanded; a second kick type (side-foot pass) with its own norms.
- **S15.** 3D lifting experiment (for example MotionBERT-style lifting), reported in a separate evaluation, before considering any 3D features.
- **S16.** A small user test: 3–5 players film themselves using the protocol. Collect qualitative feedback on how useful the output is.
