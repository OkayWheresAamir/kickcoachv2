# KickCoach v2 — MVP Specification

## Scope decisions
- **One kick type in Phase 1:** the instep (driven) kick. The literature is richest for it.
- **One capture protocol:**
  - camera side-on to the kicking direction, roughly perpendicular (within about 30°);
  - camera at about hip height, level, not moving;
  - full body and the ball visible from approach through follow-through;
  - the kicking leg nearer the camera is preferred.
- **Single player per clip.** No multi-person tracking.
- **2D only.** 3D lifting is a later experiment, not part of the MVP.

## User stories
1. As a player, I upload a clip of one kick and get back the measured variables, which ones are outside cited norms, and what to work on.
2. As a player, if my clip can't be measured reliably, I'm told **why** and how to re-film. I don't get made-up numbers.
3. As a player, I can see how my measured variables changed across my past sessions.
4. As the developer, I can run every evaluation with a single command and get reproducible results tables.

## Input
- A video file (mp4 or mov). Optional metadata: kicking foot (left/right), the known capture frame rate if it's slow motion, and the player's level (beginner, intermediate, advanced).

## Output (API response, simplified)
```json
{
  "session_id": "...",
  "quality": {
    "accepted": true,
    "reasons": [],
    "viewpoint_score": 0.86,
    "capture_fps": 240,
    "fps_trusted": true
  },
  "events": {
    "kicking_leg": "right",
    "contact_frame": 412,
    "contact_method": "ball_motion",
    "contact_confidence": 0.9
  },
  "features": [
    {
      "id": "knee_angle_contact",
      "value": 152.3,
      "unit": "deg",
      "status": "measured",
      "confidence": 0.88
    },
    {
      "id": "peak_knee_ext_velocity",
      "value": null,
      "status": "not_measured",
      "reason": "capture fps unknown"
    }
  ],
  "comparisons": [
    {
      "feature_id": "trunk_lean_contact",
      "norm_ref": "N-007",
      "verdict": "outside_norm",
      "source": "..."
    }
  ],
  "faults": [
    {
      "fault_id": "F-LEAN-BACK",
      "evidence": {"trunk_lean_contact": -14.2},
      "threshold_source": "heuristic | <citation>"
    }
  ],
  "feedback": {
    "items": [
      {
        "fault_id": "...",
        "what": "...",
        "why_it_matters": "...",
        "cue": "...",
        "drill": "...",
        "citations": ["..."]
      }
    ],
    "generated_by": "llm | template",
    "validator_passed": true
  },
  "history_delta": {"knee_angle_contact": "+6.1 vs last 3 sessions"}
}
```

## Features (Phase 1)
The rule for including a feature: it must be **important in the literature** AND **observable in side-on 2D**.

| ID | Definition | Needs | Notes |
|---|---|---|---|
| `knee_angle_contact` | Interior angle hip–knee–ankle of the kicking leg at the contact frame | Body keypoints | Core variable |
| `max_knee_flexion_backswing` | Minimum knee angle (deepest flexion) of the kicking leg between backswing start and contact | Body keypoints | |
| `trunk_lean_contact` | Angle of the line from the hip midpoint to the shoulder midpoint, relative to image vertical, at contact. Positive means forward lean | Body keypoints, level camera | Sign convention must be documented |
| `plant_foot_offset` | Horizontal distance from the plant-foot ankle (or heel/toe if available) to the ball centre at contact, divided by the player's leg length | Ball detection + foot keypoints | Unitless |
| `ankle_angle_contact` *(optional)* | Knee–ankle–toe angle of the kicking leg at contact (whether the ankle is "locked") | Foot keypoints (whole-body or MediaPipe-style model) | Include only if the chosen model has foot keypoints and E1 shows they're accurate enough |
| `segment_sequence_order` | Whether peak thigh angular velocity happens before peak shank angular velocity (proximal-to-distal) | Trusted frame rate | Report the order always. Report the lag in ms only when `fps_trusted` |
| `peak_knee_ext_velocity` | Peak knee extension angular velocity before contact (°/s) | Trusted frame rate, at least 120 (configurable) | Otherwise `not_measured` |

**Removed from v1:** hip rotation (not observable from a side view) and ankle speed as a feature (velocity reliability; it stays as a fallback signal for detecting contact).

## Comparisons (Phase 1)
- **Literature norms:** `knowledge/norms.csv`, extracted offline with LLM help, then verified by a human. The comparison is deterministic.
- **Fault catalogue:** `knowledge/faults.yaml`. Rule-based detectors. Every threshold carries a citation or the label `heuristic`.
- **Self-history:** deltas against the player's previous sessions, stored in SQLite.
- **Pro archetypes: removed** from the main path. They may stay as a disabled legacy module.

## Out of scope (Phase 1)
- Kick types other than the instep kick
- Multi-camera setups or 3D
- Injury or medical advice
- Accounts and authentication
- A mobile app
- Deployment (Phase 2)
- Real-time processing
- Spin or trajectory of the ball

## Acceptance criteria (Phase 1, the resume version)
1. `python -m eval.run_all` reproduces E1–E5 and E8 from committed labels, producing `results/*.csv` and a summary markdown file.
2. The pose model is chosen by the E1 evidence, with the decision recorded in `docs/DECISIONS.md`.
3. The pipeline never outputs a value that breaks a sanity bound. Such values are flagged `not_measured`, with the reason.
4. Clips that fail the quality gate return reasons, with no feature values.
5. Velocities appear only when `fps_trusted` is true.
6. The feedback validator rejects any LLM output that cites an unknown source ID or states a number that differs from the measured value beyond tolerance. On rejection, the system falls back to a template.
7. The API runs with no LLM key. Feedback comes from templates.
8. The README reports the actual evaluation numbers, the dataset size, the labelling protocol, how consistent Aamir's own labels are (labelling repeatability), and its limitations.
