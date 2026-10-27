# KickCoach v2 — Architecture and Decisions

## Data flow
```
video
 └─ ingest            read capture/playback fps (ffprobe), resolution, frames      [deterministic]
     └─ pose          model adapter -> canonical skeleton (x, y, conf) per frame    [pretrained model]
         └─ signal    gap fill -> L/R leg-swap fix -> low-pass filter -> derivatives [deterministic]
             └─ events    kicking leg, ball track, contact frame, backswing         [pretrained detector + deterministic]
                 └─ quality   viewpoint score, visibility, fps trust -> accept/refuse [deterministic]
                     └─ features  definitions + sanity bounds                       [deterministic]
                         └─ compare   norms table + fault rules + self-history      [deterministic]
                             └─ feedback  LLM structured output -> validator -> or template [LLM + deterministic]
                                 └─ storage  SQLite session + full intermediate record
```
The FastAPI layer wraps this pipeline. The frontend from v1 can be reused at first.

## Key decisions
Record each one in `docs/DECISIONS.md` as you implement it, with the evidence.

**D1. The LLM never measures.**
Measurement is a pretrained pose model plus deterministic geometry. The LLM writes feedback and assists offline extraction of norms.
*Rejected:* asking a vision LLM for angles. That's the baseline we evaluate (E4), not the system.

**D2. Ground truth is human-labelled.**
A tool pre-fills keypoints from the best available model. Aamir corrects and confirms them, and marks the contact frame. A repeat-labelling subset measures his own consistency.
*Rejected:* LLM labels (circular) and literature PDFs as labels (they give norms, not per-frame truth).

**D3. A feature exists only if it matters in the literature AND can be observed in side-on 2D.**
Hip rotation is dropped. See `02_MVP_SPEC.md` for the feature table.

**D4. Capture protocol plus a quality gate, instead of 3D.**
Monocular 3D models perform poorly on athletic motion unless fine-tuned. In AthletePose3D, error fell from 214 mm to 65 mm only after fine-tuning. Phase 1 therefore uses 2D and refuses clips with a bad viewpoint.
- **Viewpoint score heuristic:** the ratio of apparent shoulder or hip width to torso length. In a true side view, that width is small. Calibrate the threshold on labelled clips Aamir marks as good or bad; it is a heuristic.

**D5. The pose model is chosen by benchmark (E1), not by reputation.**
- **Candidates, kept to three for day 1:**
  - MoveNet (the v1 baseline);
  - RTMPose via `rtmlib`, preferably a variant with foot keypoints (Halpe-26 body-with-feet or whole-body; check what rtmlib currently provides);
  - one of MediaPipe Pose (has heel and foot-index landmarks), YOLO11-pose, or ViTPose. Choose whichever installs cleanly.
- **Prior evidence:** a side-view kickboxing study preferred ViTPose-L and RTMPose-L over MoveNet and YOLO11x-Pose for accuracy and keypoint completeness.
- **Interface:** `estimate(frames) -> array[T, K, 3]` in a canonical skeleton (shoulders, hips, knees, ankles, plus heel and toe if available), with per-model mapping tables.

**D6. Frame rate handling.**
- Read the stream's frame rate with ffprobe.
- Internet slow-motion clips have **playback fps ≠ capture fps**, so treat the capture rate as unknown unless the manifest gives it.
- `fps_trusted = capture rate known AND at least 120` (configurable).
- Angles are always computed. Velocities, and timing in ms, only when `fps_trusted`.
- *Why:* foot speed near impact is very high. At 30 fps the foot moves tens of centimetres between frames.

**D7. Filter before differentiating.**
- Fill gaps where keypoint confidence is low (bounded interpolation), then apply a zero-lag low-pass Butterworth filter (`scipy.signal.butter` + `filtfilt`).
- Pick the cutoff relative to the capture rate. Document the choice, and test sensitivity on the evaluation set. A Savitzky–Golay filter is the alternative.
- Compute derivatives by central differences on the filtered signal.
- *Why:* differentiating noisy keypoints amplifies noise. This is the likely cause of v1's 16,444°/s reading.

**D8. Fix left/right leg swaps.**
In a side view, models often swap the near and far leg. Per frame, choose the left/right assignment that minimises frame-to-frame keypoint displacement. Identify the kicking leg as the one with the larger peak foot displacement before contact, unless the user supplies it.

**D9. Contact detection is ball-based first.**
- Track the ball with a COCO-trained detector (for example, the ultralytics YOLO "sports ball" class). Contact is the first frame after which the ball's displacement exceeds a threshold, within a window around the foot's approach.
- *Fallback:* the frame of peak kicking-foot speed.
- Always report `contact_method` and a confidence. E3 compares the two methods.

**D10. Sanity bounds come from the literature and flag values instead of clamping them.**
- *Example:* Kellis & Katis report knee extension velocities of about 860–1720°/s, and shank angular velocity up to about 1891°/s. Set the bound with a margin, cite it, and flag anything beyond it as `not_measured` with a reason.
- Angles must lie within anatomical ranges (cite or mark `heuristic`).

**D11. Comparison = verified norms + fault rules + self-history. Pro archetypes are dropped.**
- Pros don't make beginner mistakes, archetypes need lots of body-type data, and the v1 archetypes came from an unvalidated pipeline.
- Literature caveats to show in the user interface:
  - most studies use adult male players in labs with 3D motion capture;
  - accurate kicks use slower motion than maximum-power kicks.

**D12. Norms are extracted offline and verified, not retrieved at runtime.**
- An LLM extracts candidate rows from 2–4 open-access papers into `knowledge/norms.csv`, using a Pydantic schema:
  `norm_id, variable, kick_type, population(level, age_group, sex), stat(mean, sd, min, max), n, method(3D mocap / 2D video), source, page_or_table, notes, verified(bool)`
- Aamir verifies every row before `verified=true`. Only verified rows are used at runtime.
- *Why:* numbers that drive verdicts must be stable and auditable. RAG at runtime would make them vary from request to request.
- E5 measures how accurate the extraction was.

**D13. LLM feedback uses structured output plus a deterministic validator.**
- **Input to the LLM:** measured features, faults with evidence, the matched norm rows, the player's level, and (in Phase 2) retrieved coaching passages with IDs.
- **Output:** a Pydantic model with items `{fault_id, what, why_it_matters, cue, drill, citations[]}`.
- **Validator:**
  - schema-valid;
  - every citation ID is in the provided set;
  - every number in the text matches a measured value within tolerance;
  - no claims about features marked `not_measured`.
- On failure: retry once with the validator's errors, then fall back to the template. Log both attempts.

**D14. RAG is used for coaching drills and cues only (Phase 2).**
- A small curated corpus: about 30–60 chunks, each tagged with `fault_ids, kick_type, level, source_url`.
- Retrieval: metadata filter on the fault, then embedding similarity.
- **Baseline:** metadata filter only. With a small corpus, the filter alone may be enough. E6 decides, and the README reports the result honestly.
- Embeddings: Gemini embeddings or a local sentence-transformers model. Pick one and record why.
- LangChain only if its retriever abstractions genuinely save code. A plain vector store (Chroma, or numpy plus cosine similarity) is acceptable.

**D15. No agents and no LangGraph.**
The pipeline is linear. It doesn't branch on intermediate results, wait for a human, or call tools. Adding an orchestration framework would be decoration. *In short:* "I left out agents because nothing in the flow required them."

**D16. SQLite replaces CSV.**
- Tables: `sessions`, `features`, `faults`, `feedback`, `artifacts`.
- `artifacts` holds the paths to stored intermediate arrays, so any output can be reproduced.

**D17. LLM provider.**
Keep Gemini (it's the existing choice and is low-cost) behind a small provider interface. Check current model names and the video-input API in the google-genai documentation before use.

## Risks and constraints
| Risk | Mitigation |
|---|---|
| Internet clips have unknown capture frame rate, compression, or bad angles | Manifest records what's known; `fps_trusted` gating; quality gate; prefer clean side-on clips when curating |
| Too few usable clips | Target 20–30. Fewer is acceptable if reported honestly; the evaluation reports n |
| Labelling time | Pre-fill from a model; about 3 frames per clip; time-box it |
| Pose models fail on blurred frames at contact | E1 reports contact-frame error separately; the quality gate uses keypoint confidence |
| Ball not detected (small, blurred, occluded) | Fall back to foot speed; report the method; E3 measures both |
| Norms that don't fit 2D side-view measurement | The `method` column records it; only comparable variables are used; caveats shown |
| Copyright and privacy of clips | Clips and frames are never committed or published; only URLs and timestamps. Prefer adult players. Use only frames in evaluation, never in a public demo, without rights |
| Coaching corpus licensing (Phase 2) | Prefer open-access sources; store source URLs; don't republish full text |
| API cost | Gemini baseline runs once per clip (3 times only if budget allows, for variance); cache responses to disk |
| LLM unavailable | Template feedback; the pipeline still runs |
