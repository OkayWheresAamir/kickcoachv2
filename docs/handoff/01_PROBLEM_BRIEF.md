# KickCoach v2 — Problem Brief

## One line
KickCoach measures a footballer's kick from an ordinary phone video. It compares the measurements against norms from the biomechanics literature and the player's own past kicks, and explains the specific faults with sourced coaching cues. Every measurement is validated against human-labelled ground truth.

## Origin
KickCoach won **3rd place at the Synapse AI Hackathon (DTU, Feb 2026)**. Aamir led the team, designed the pipeline, and sourced the reference dataset.

v1 worked as a demo, but its measurements were unvalidated:
- the frame rate was assumed;
- it produced physically impossible velocities;
- one feature (hip rotation) can't be observed from the camera angle used;
- players were compared against "pro archetypes" built with the same unvalidated pipeline.

v2 is the engineering rebuild.

## Target user
An amateur or developing player without regular access to a coach. They have watched technique tutorials but can't see what *their own* kick does differently.

## The gap
- **YouTube and tutorials** show the ideal, but can't tell you how *your* kick differs from it.
- **A coach or experienced friend** is better and more fun, but isn't always available, and rarely gives measured or consistent feedback over time.
- **A general-purpose AI model** (upload a video to Gemini or ChatGPT and ask "how's my technique?"):
  - gives fluent, qualitative advice;
  - *Hypothesis to test:* it cannot reliably **measure** joint angles or the moment of contact, and may state numbers it didn't measure;
  - it has no memory of your previous sessions.
- **KickCoach v2:**
  - measures specific variables;
  - says when it *can't* measure something (bad camera angle, frame rate too low);
  - compares against cited norms and your own history;
  - points to sourced drills.

## What we claim, and what we don't
- **We claim:** approximate 2D measurement of selected kick variables from side-on video, with the measurement error quantified against human labels and compared with a frontier-model baseline.
- **We don't claim:**
  - lab-grade or 3D biomechanics;
  - injury assessment;
  - personalisation beyond what the cited literature supports;
  - validity for camera angles outside the capture protocol.

The literature itself cautions that 2D video analysis of fast sports movements gives approximate feedback, not precise individual-level measurement.

## Why this is AI engineering, not a wrapper
It passes the structural tests we used to choose projects:

1. **It needs proof against ground truth.** Measurements are checked against human labels, and the frontier model is evaluated on the same labels.
2. **It holds state a chat lacks.** Session history lets the system track whether a fault is improving.
3. **It gets better as models improve, not obsolete.** Better pose models plug into the same benchmark, and better LLMs write better feedback. The value lives in measurement validity, the evaluations, and the curated knowledge, not in a gap in model capability.
4. **It's honest about what's observable.** Variables that can't be seen from a side-on 2D view are excluded, and unmeasurable clips are refused.

## Assumptions and open hypotheses
| # | Assumption | Status | How it gets tested |
|---|---|---|---|
| H1 | A frontier video model misestimates knee angle and trunk lean at contact more than the pose pipeline does | Hypothesis | Eval E4 |
| H2 | ViTPose or RTMPose beat MoveNet on fast side-view kicking frames | Supported by a related study on kickboxing kicks; not yet tested on football | Eval E1 |
| H3 | Ball-based contact detection beats peak ankle speed | Hypothesis | Eval E3 |
| H4 | Velocity features are unreliable from internet slow-motion clips with unknown capture frame rate | Strongly expected | Velocities are reported only when the capture rate is trusted |
| H5 | Players find measured, cited feedback more useful than generic tips | Not validated; no user study in Phase 1 | Future work |

## Key sources (verify, then cite in `knowledge/sources.md`)
- Kellis & Katis (2007), *Biomechanical Characteristics and Determinants of Instep Soccer Kick*, Journal of Sports Science and Medicine. Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC3786235
- Validation of Pose Estimation Models for Automated Kickboxing Technique Analysis. Compares MoveNet, YOLO11x-Pose, ViTPose-L and RTMPose-L on side-view kicks: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13667236/
- AthletePose3D (CVPR 2025 Workshops). Monocular pose models show strong angle correlation but limited velocity accuracy on athletic motion: https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Yeung_AthletePose3D_A_Benchmark_Dataset_for_3D_Human_Pose_Estimation_and_CVPRW_2025_paper.html
