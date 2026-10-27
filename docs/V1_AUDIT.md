# v1 audit

Audit of the hackathon version, done 2026-10-27 (Phase 0).

- **Code audited:** `kickcoach-2` @ `deb3c4e` (the continuation of `kickcoach-synapse.ai`), at `../kickcoach-2`.
- **Also read:** the original repo's notebooks at github.com/OkayWheresAamir/kickcoach-synapse.ai.
- **Evidence:** copies of v1's two CSVs are in `tools/audit_v1/data/`. Every claim marked *verified* can be reproduced with the scripts named next to it.

## 1. How v1 actually works

```
POST /feedback (app.py)
 └─ video saved to a temp file
 └─ pose_engine.analyze_video_file
     ├─ fps = OpenCV CAP_PROP_FPS (container rate), or 30 if OpenCV returns 0
     ├─ per frame: tf.image.resize_with_pad -> 192x192 -> MoveNet (3.tflite, 17 COCO keypoints)
     │   └─ keypoints mapped back as (y_norm*h, x_norm*w)          <- bug 2.2
     ├─ trunk tilt   = hip-mid -> shoulder-mid angle vs image vertical (sign = image left/right)
     ├─ hip rotation = signed angle between the 2D shoulder line and hip line
     ├─ per leg: knee angle (hip-knee-ankle); ankle speed = px displacement * fps;
     │           knee angular velocity = (angle change since last valid frame) * fps   <- bug 2.4
     ├─ causal EMA (alpha 0.35) on the angles; velocities left raw
     ├─ a "kick candidate" = ankle > 120 px/s OR |knee vel| > 80 deg/s, 0.4 s cooldown,
     │   stop after 10 candidates                                    <- bug 2.5
     └─ best kick = candidate with the highest ankle speed           <- bug 2.3
 └─ feedback_engine.generate_feedback
     ├─ select_profile(age, level, ...) -> one of 6 hard-coded "pro" profiles  <- finding 2.6
     ├─ % deviation per feature vs that profile; unsourced severity thresholds
     ├─ deviations SORTED BY SEVERITY                                <- bug 2.1
     └─ plain-text prompt -> Gemini 2.5 Flash REST (key in URL query) -> JSON repair heuristics
         (no schema validation; with no key the endpoint returns 503, so there is no fallback)
 └─ app._append_history -> kick_history.csv                          <- bug 2.1
```

## 2. Findings

### 2.1 The "16,444°/s knee velocity" is a CSV bug, not a measurement (*verified*)
- **Mechanism:**
  - `feedback_engine._compute_deviations` sorts features by severity, so the order changes from session to session.
  - `app._append_history` builds each row in that order and writes it with `csv.DictWriter(fieldnames=list(record.keys()))`.
  - The header is written only for the first session. Every later row is therefore written in its own order under the first row's header.
- **Evidence:** `tools/audit_v1/reconstruct_history_scramble.py` tries every feature assignment for each row and keeps only those that reproduce v1's sort order. For the frame-19 row exactly one assignment survives:
  - knee angle 110.6°, trunk 31.9°, hip rotation −15.1°;
  - **ankle speed 16,444 px/s** (written under the `knee_ang_vel_dps` header);
  - **knee velocity −1,124°/s**.
  - Rows 71 and 21 are scrambled too. Row 71 matches one of v1's own pipeline outputs in `final_kick_features.csv`.
- **Consequence:** the handoff blamed the value on differentiating noisy keypoints (D7). Filtering before differentiating is still correct, for the reasons in 2.4, but it doesn't explain this row. A 16,444 px/s ankle speed is itself suspicious: at 30 fps it is about 550 px per frame, most likely a keypoint jump (2.3).
- **In short:** "The famous impossible reading turned out to be a data-integrity bug. Writing rows in a severity-sorted order under a fixed CSV header put an ankle speed in px/s into the knee-velocity column. I proved it by brute-forcing which feature assignments are consistent with the sort."

### 2.2 Every v1 angle is geometrically distorted (*verified in simulation, and on real clips in Phase 1*)
- **Mechanism:**
  - `resize_with_pad` shrinks the frame to fit 192×192 and pads the short axis.
  - MoveNet's coordinates are normalised to that padded square.
  - `scaled_keypoints_from_output` multiplies them by the original height and width, ignoring the padding. On 16:9 video the vertical axis is compressed to 56%; portrait video compresses the horizontal axis.
- **Evidence** (`tools/audit_v1/letterbox_distortion.py`; regression test `tests/test_letterbox.py`):

  | Pose | True | v1, landscape 16:9 | v1, portrait 9:16 |
  |---|---|---|---|
  | Knee angle | 120° | 108.0° | 135.7° |
  | Knee angle | 150° | 134.3° | 162.0° |
  | Knee angle | 170° | 162.6° | 174.3° |
  | Trunk lean | 15° | 25.5° | 8.6° |

- **Real clips (Phase 1, `tools/audit_v1/real_clip_followups.py` → `results/audit/phase1_followups.md`):** v1's own `3.tflite` on 6 full source videos (4:3 and 16:9). v1's mapping minus the corrected one, over every knee angle with all three points at score ≥ 0.2: median −4.8° to −12.0° per clip, 90th-percentile absolute difference 11–25°. The sign matches the simulation (landscape video reads the knee as more bent). The corrected keypoints were checked by eye on a 1080p frame and sit on the joints.
- **Fix:** `kickcoach/pose/letterbox.py` (tested round-trip). Every adapter must map through an explicit, tested transform.
- **In short:** "The hackathon version's keypoint un-padding ignored the letterbox, so on landscape video it read a 150° knee as 134°. It was a systematic bias, not noise, and only a test with known geometry catches it."

### 2.3 Kick selection prefers glitches (*code reading*)
- **Mechanism:** the "best" kick is the candidate with the highest ankle speed. A keypoint that jumps between frames, or swaps legs, produces the highest apparent speed, so the selection rule actively picks corrupted frames.
- **Fix:** ball-based contact detection, with peak foot speed computed on filtered signals after the leg-swap fix (handoff D8, D9).

### 2.4 Velocities are unfiltered, and gaps inflate them (*code reading*)
- **Mechanism:** knee angular velocity is computed as `(angle - prev_valid_angle) * fps`. The previous angle is only updated on frames where the angle could be computed. After a gap of N frames, the change over N frames is therefore divided by a single frame's time, inflating the velocity by up to N×.
- **Also:** the EMA smoothing is causal (it lags), so the reported angles at the "kick" frame describe earlier frames.
- **Fix:** handoff D7 (bounded gap fill, zero-lag Butterworth `filtfilt`, central differences).

### 2.5 The real kick may never be analysed in longer clips (*superseded by 2.8 in Phase 1*)
- **Mechanism:**
  - The thresholds are low: 120 px/s is 4 px per frame at 30 fps, so ordinary run-up motion triggers them.
  - The scan stops after 10 candidates, and the cooldown between candidates is 0.4 s.
  - In a clip with more than about 4–5 s of motion before the strike, the loop can therefore stop before reaching the kick.
- **Also:** the thresholds are in px/s, so they depend on resolution and camera distance.
- **Phase 1 result:** as written, the candidate loop never fires at all (2.8), so the cap is never reached. In a counterfactual replay with 2.8 fixed, the 10-candidate cap was used up before the kick window started on 5 of 6 real clips, so the concern was right.

### 2.6 The "pro reference profiles" are not sourced (*verified*)
- **"Elite Power Striker":** its reference values equal, to two decimal places, v1's own output for one clip (`final_kick_features.csv`, frame 185: knee 139.60, trunk 10.37, hip −4.09, ankle 2227.17, knee velocity 1689.65).
  - The hackathon notebook's `VIDEO_PATH` is `cr7.mp4`, so this was probably a single pro clip. That is an inference.
- **The other five profiles:** they look hand-entered.
- **The code comment:** it says "Values sourced from published biomechanics literature and motion-capture data", which is not the case.
- **Scale of v1's data:** `final_kick_features.csv` holds only 3 distinct outputs across 61 rows (frame 185 ×39, frame 71 ×14, frame 26 ×8), so the demo was run repeatedly on the same 2–3 clips.
- **Compounding problems:**
  - the reference values carry the 2.2 distortion;
  - ankle speed in px/s can't be compared across videos with different resolution or camera distance;
  - a % deviation from near-zero signed references is meaningless. A hip rotation of −90° against −2.5° is a 3,500% deviation.
- **Seed for v2:** the original repo's `dataset-YT-links.ipynb` lists 12 unique YouTube links, 15 entries with 3 duplicates (corrected in Phase 1). These seed the v2 manifest (Phase 1).

### 2.7 Smaller issues (*code reading*)
- **"Hip rotation" can't be measured from the side.** Both the shoulder line and the hip line are foreshortened to near-zero length, which is where the −90° values come from. This confirms handoff D3.
- **Trunk-tilt sign** follows image left/right, not the kick direction. A player kicking leftwards gets the opposite sign, yet is compared with the same +10° reference. Fixed in `kickcoach/features/geometry.py` with an explicit `facing`.
- **Minor:**
  - `_detect_kicking_leg` is dead code;
  - the Gemini key is sent as a URL query parameter, so it ends up in logs;
  - CORS is `*` with credentials;
  - `app.py`'s header still says `routes.py`.
- **Confirmed in Phase 1:** `3.tflite` takes a float32 [1, 192, 192, 3] input and returns [1, 1, 17, 3], i.e. MoveNet SinglePose Lightning (Thunder takes 256).

### 2.8 The kick-candidate gate can never fire (*verified on real clips, Phase 1*)
- **Mechanism:** in `analyze_video_file`, the per-leg update stores this frame's ankle position and knee angle as `prev_ankle` / `prev_knee_angle`. The detection block a few lines later recomputes ankle speed and knee angular velocity against those just-overwritten values, so both are always 0 and neither threshold is ever crossed.
- **Consequence:** every upload falls through to `_fallback_peak_frame`, which returns the single frame with the highest one-step ankle speed in the whole video (no cooldown, no cap), the most glitch-prone choice possible (2.3).
- **Evidence:** a replay of the loop on v1-style MoveNet output from 6 full source videos: 0 candidates on all 6; the fallback frame fell inside the kick window on 2 of 6 (`results/audit/v1_candidates.csv`).
- **In short:** "The hackathon version's kick detector never actually ran; a variable was overwritten before it was compared, so every result came from a fallback that picked the fastest-moving frame, often not the kick."

### 2.9 v1 fed MoveNet BGR frames (*verified on real clips, Phase 1*)
- **Mechanism:** frames come from OpenCV (`cv2.VideoCapture`, BGR order) and go straight into MoveNet, which expects RGB.
- **Evidence:** on the same 6 clips, the mean keypoint score is lower with BGR input on every clip (by 0.01–0.08), and keypoints move by a median 0.04–0.16 torso lengths.
- **Fix:** v2 decodes with PyAV to RGB and converts only where a library documents BGR input (rtmlib).

### What is worth porting
- the MoveNet model and its inference loop (with the 2.2 fix), as the E1 baseline;
- the drill texts in `reference_data.DRILL_LIBRARY`, as unsourced seed material for the template fallback (they must be marked unsourced until checked);
- the `index.html` frontend, as the first UI (handoff: "reuse at first").

## 3. Conflicts

### 3.1 July 2026 plan (`kickcoach-2/docs/PROJECT_STATE.md`, `CLAUDE.md`, `CLAUDE.local.md`) vs the handoff

The handoff wins on scope.

| Topic | July plan | Handoff (adopted) |
|---|---|---|
| Sequence | Shippable backend → React app → analysis quality → accounts → PWA | Evaluation first. Deployment and UI come after the resume version |
| Comparison | "vs pros AND vs your own past kicks" is core | Pro archetypes dropped. Cited norms plus self-history instead (and 2.6 shows the archetypes were unsourced) |
| Features | The 5 v1 features are a "shared contract", hip rotation included | Hip rotation removed. Ankle speed is used only as a contact-detection fallback |
| Frontend | Mobile-first React PWA | Reuse the v1 page at first |
| Storage | SQLite + anonymous ID (Phase 2), Postgres + auth (Phase 4) | SQLite, no accounts. Compatible |
| Open questions | requirements.txt vs pyproject+uv; LFS vs download script for weights | Resolved: pyproject + uv; weights downloaded by script, never committed |

### 3.2 Handoff vs evidence
- **"The frame rate is assumed instead of read":** only partly true. v1 reads the container rate through OpenCV and falls back to 30 only if that fails. The real flaw is that the container (playback) rate is not the capture rate for slow-motion clips.
- **"Likely cause" of 16,444°/s:** wrong. See 2.1.
- **Missing from the handoff's list of known v1 problems:** the letterbox distortion (2.2, the largest systematic error), the max-speed selection bias (2.3), and the 10-candidate cut-off (2.5).
- **Phase-1 cut priority:** the handoff (S0–S7, S10, then S9 template-only, then S8) drops the LLM components first, which works against the project's goal of showing evaluated AI engineering. Changed in DEC-002.

## 4. What v1 claimed
- **v1's headline claim:** "compares them with pro reference profiles built from a curated YouTube dataset". It doesn't survive scrutiny, given 2.6. v2 replaces it with measured, evaluated claims.
- **Still true:** the 3rd-place result and team-lead role. They stay.
- **The audit is worth telling in its own right.** Finding 2.1 (a data-integrity bug disguised as a physics problem) and 2.2 (a systematic geometric bias) are good "how it was debugged" stories.
