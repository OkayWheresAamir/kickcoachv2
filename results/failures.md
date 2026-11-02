# Failure analysis: the worst cases (Phase 4)

The images are video frames, so they stay local (`data/failures/`, gitignored). Regenerate
them with `uv run python -m eval.failures`. In each panel, green = Aamir's labelled points,
orange = the system's points on that frame, and the kicking leg is drawn thicker. The
diagnoses below are Claude's, written from the images and the per-clip numbers. They are
not labels.

## E2: angles (v2 end to end, largest absolute errors)

| case | error | image | diagnosis |
|---|---|---|---|
| c06 max knee flexion | −28.4° | `e2/c06_max_knee_flexion_backswing.png` | **Wrong contact.** No resting ball is found (the player is small; the detector's best score is 0.22, DEC-018), so contact falls back to foot speed and lands 17 frames late, in the follow-through. The backswing found before it is the raised follow-through leg, which RTMPose garbles (the knee–ankle segment zig-zags). |
| c01 knee at contact | +16.5° | `e2/c01_knee_angle_contact.png` | **The bracket shift overshoots** (DEC-022). The uncorrected pick (60) was exact; +2 frames lands on 62, and on the pipeline's series the knee straightens 10.5° between those frames. It's the one tuning bracket clip the shift makes worse. |
| c06 trunk lean | +15.6° | `e2/c06_trunk_lean_contact.png` | The same wrong contact as above: at frame 91 the trunk is leaning back in the follow-through. |
| c08 max knee flexion | −10.3° | `e2/c08_max_knee_flexion_backswing.png` | **A frame-choice disagreement, not a keypoint error.** At the pipeline's frame 56 the kicking foot is visibly higher behind the body than at the labelled frame 48. RTMPose's knee angle is 78.8° there vs 88.3° at frame 48, and the labelled points at 48 agree with RTMPose (measurement part −0.8°). Raised with Aamir as a QA question, not edited (DEC-007). |

Also notable: c14's max flexion (−9.7° event part) is the 4-frame partial leg swap near
contact that Phase 3 found (JOURNAL Phase 3 §27), and c11's +8.7° measurement part is a
held-out clip where RTMPose's backswing knee sits off the labelled one.

## E3: contact (v2 as shipped, every clip with a nonzero error)

| case | error (frames) | image | diagnosis |
|---|---|---|---|
| c06 | +17 | `e3/c06_contact.png` | The ball sits under the feet at frames 73–74, but the detector never finds it at rest. The foot-speed fallback picks the follow-through speed peak at 91. |
| c01 | +2 | `e3/c01_contact.png` | The bracket shift overshoots; the uncorrected pick was exact (see E2). |
| c13 | −2 | `e3/c13_contact.png` | The ball is hidden or blurred under the foot for 42 frames (bracket 58–100). At the stated 300 fps, −2 frames is 6.7 ms. |
| c15 (held-out) | +1 | `e3/c15_contact.png` | Frames 74 and 75 both show the foot on the ball: the one-frame ambiguity DEC-015 expects between labellers. |

## E8: reliability

| case | what | image | diagnosis |
|---|---|---|---|
| c25, v1 | knee velocity 3,963°/s | `e8/c25_v1_velocity.png` | v1's skeleton is squashed vertically (the letterbox bug, V1_AUDIT 2.2), and on frame 68 one leg's knee–ankle reaches across to the other leg. One frame's angle jump × 30 fps is the "velocity". |
| c13, v1 | knee velocity −3,754°/s | `e8/c13_v1_velocity.png` | On frame 62, 26 frames before contact (it's the backswing, not the kick), v1 puts the kicking ankle on the plant ankle. The knee "angle" collapses about 125° in one frame. |
| c23, gate | refused correctly, ratio 0.564 | `e8/c23_gate.png` | A rear-oblique view, so both shoulders show. It's the not-OK clip nearest the 0.5 threshold. |
| c06, gate | accepted correctly, ratio 0.437 | `e8/c06_gate.png` | The highest side-on ratio. It's measured around the wrong (follow-through) contact, where the trunk turns toward the camera, so a contact error also moves the gate's window. |

## E4: Gemini (partial: 8 of 19 scored clips asked so far; see `results/e4_summary.md`)

| case | error | image | diagnosis |
|---|---|---|---|
| c02 knee at contact | +40.5° | `e4/c02_knee.png` | **Right moment, wrong angle.** Gemini's contact is 1 frame from the label, yet it reports 161° for a knee that is visibly bent (120.5° from the labels). |
| c18 knee at contact | +33.7° | `e4/c18_knee.png` | The same pattern: contact +1 frame, a clearly flexed knee (122.3°), answered as 156°. |
| c03 knee at contact | +33.5° | `e4/c03_knee.png` | The same again (held-out): contact −1 frame, 117.5° answered as 151°. |
| c01 (no numbers) | – | – | Gemini answered that the video "is completely static and consists of a single frozen frame repeated". The file isn't frozen: it decodes to 150 distinct frames, and Gemini's own usage shows 120 sampled frames (31,920 video tokens). Same container settings as clips it answered (one keyframe, pts from 0). Queued for a repeat call. |

Across the clips it answered, Gemini's knee angles at contact sit between 138° and 161°
whatever the frame shows. They don't move with the labelled angles (Pearson r −0.42 on 7
clips; the pipeline's is +0.88). It reads like a prior for "a nearly straight leg at impact",
not a measurement of this kick. Its contact *timing* is good: within ±1 frame on 6 of 7.
