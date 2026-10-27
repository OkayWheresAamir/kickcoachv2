# KickCoach v2 — Evaluation Plan

The evaluation is the core of this project. Build it alongside the pipeline, not at the end. Every number in the README must come from these scripts.

## Dataset
- **Source:** 20–30 public internet clips of instep kicks. Aamir approves the manifest. Clips are downloaded locally and never committed.
- **Curation criteria:**
  - roughly side-on;
  - full body and ball visible;
  - single kicker;
  - instep or driven kick;
  - a mix of skill levels;
  - prefer adult players;
  - prefer clips with a known capture frame rate (stated in the description, or real-time footage at a high frame rate).
- **`data/manifest.csv` columns:**
  `clip_id, url, start_s, end_s, playback_fps, capture_fps_known (y/n), capture_fps, slow_motion (y/n), kicking_leg, level_guess, view_ok_human (y/n), notes`
- **Split:** keep about 5 clips as a held-out set that is never used to tune thresholds, filter cutoffs, or the viewpoint gate. Report results on both the tuning and held-out sets.

## Ground truth (human-labelled by Aamir)
- **Per clip:**
  - the contact frame index (Aamir decides by eye);
  - the frame of deepest backswing knee flexion;
  - keypoints on about 3 frames: contact, contact −1, and maximum backswing.
- **Keypoints:** both shoulders, hips, knees and ankles; heel and toe of both feet if visible; the ball's centre.
- **Tool:** `tools/label_tool.py` pre-fills keypoints from the best available model. Aamir drags each point to the correct position and confirms. Unconfirmed points are not ground truth.
- **Labelling repeatability:** Aamir relabels about 10 frames a second time, on a different occasion, without looking at his first labels. Report his own pixel and angle variation. **This sets the floor:** no model can be shown to be more accurate than the labeller is consistent.
- **Derived ground-truth angles** are computed from the labelled keypoints with the same geometry code the pipeline uses. Unit-test that code separately.

## Evals
| ID | Question | Metric(s) | Compared |
|---|---|---|---|
| **E1 Pose accuracy** | Which pose model places keypoints best on kicking frames? | Keypoint error normalised by thigh length; PCK@0.1 thigh length; per joint, with kicking-leg joints and contact frames reported separately; missing-keypoint rate | MoveNet vs RTMPose variant vs third model |
| **E2 Angle accuracy** | How close are the pipeline's angles to angles from the labels? | MAE and mean signed error (bias) in degrees for `knee_angle_contact`, `max_knee_flexion_backswing`, `trunk_lean_contact` (and `ankle_angle_contact` if used) | v1-style pipeline (MoveNet, no filtering) vs v2 pipeline |
| **E3 Contact detection** | Does ball-based contact beat peak foot speed? | Absolute frame error; % within ±1 frame; % within ±2 frames; detection failure rate | Ball-motion method vs peak-foot-speed method |
| **E4 Frontier baseline** | Can a frontier video model measure these angles? | Same MAE and bias as E2, plus contact-time error; rate of refusals or "can't determine" responses; rate of numbers stated without hedging | Gemini (video input, structured-output prompt asking for knee angle at contact, trunk lean at contact, contact timestamp) vs v2 pipeline |
| **E5 Norm extraction** | How accurate is LLM-assisted extraction of norms from papers? | Field-level accuracy against Aamir's verified rows; hallucinated-row rate (rows with no support in the source) | LLM extraction vs human verification |
| **E6 Retrieval** (Phase 2) | Does retrieval fetch the right coaching passage for a fault? | Recall@3 and MRR on a hand-made fault-to-relevant-chunk mapping | Metadata filter only vs filter + embeddings |
| **E7 Feedback faithfulness** (Phase 2) | Is the LLM's feedback grounded in what was measured and retrieved? | Validator pass rate on the first try; failure reasons; Aamir's 1–5 rubric on 10 outputs (correctness, specificity, actionability) | LLM vs template |
| **E8 Reliability** | Does v2 eliminate impossible outputs and refuse bad clips? | Count of values beyond sanity bounds (v1 vs v2); precision and recall of the quality gate against Aamir's `view_ok_human`; share of clips where velocities were correctly withheld | v1 vs v2 |

## E4 protocol details (the headline comparison)
- Same clips, same ground truth.
- The prompt tells the model the kicking leg and asks for numbers in a fixed JSON schema. It allows "cannot determine".
- Run at temperature 0 if supported. Cache raw responses in `results/e4_raw/`.
- Report:
  - the error on clips where the model answered;
  - how often it answered at all;
  - for both the model and the pipeline, the error on clips where *both* answered.
- **Report whatever comes out.** If Gemini is competitive, that is a valid finding: state it plainly and adjust the project's claims.

## Unit tests (deterministic code)
- **Geometry:** angle functions on hand-computed cases, including 0°, 90°, 180°, and degenerate cases where points coincide.
- **Synthetic kinematics:** generate a stick-figure kick with known joint angle curves and a known contact frame. Add noise. Check that filtering plus the features recover the known values within tolerance at several frame rates. This is the best test of the signal and feature code.
- **Leg-swap fix:** a synthetic sequence with injected swaps is repaired.
- **Sanity bounds:** values beyond the bounds become `not_measured`, never clamped numbers.
- **Validator:** cases covering a bad citation ID, a mismatched number, and a claim about an unmeasured feature are all rejected.
- **API:** runs with no key and returns template feedback.

## Reporting
- `results/summary.md` is generated by `eval/run_all.py`. It includes:
  - the dataset n (tuning and held-out);
  - the labelling repeatability floor;
  - a table per evaluation;
  - a short failure analysis: the 3–5 worst cases, with a frame image saved locally and a one-line diagnosis each.
- The README includes the summary, plus a "Limitations" section: 2D, small n, internet clips, literature population bias, no user study yet.

## Resume claims
Write these only after running the evaluations. Template:
> *Rebuilt KickCoach's measurement pipeline and evaluated it on N hand-labelled kick frames. The chosen pose model reduced knee-angle error from A° to B° versus the hackathon version. A frontier video model measured the same angles with C° error. Ball-based contact detection, low-pass filtering and literature-based sanity bounds eliminated physically impossible readings. Feedback is grounded in verified literature norms through structured LLM output with a deterministic validator.*

Add "RAG over a curated coaching corpus (recall@3 = X)" only after E6 has run.
