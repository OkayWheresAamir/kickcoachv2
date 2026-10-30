"""Quality gate (handoff D4): measure only clips filmed close to the capture protocol; refuse
the rest with reasons and no feature values (MVP acceptance criterion 4).

Checks, over the kick window (backswing valley start to contact):
- viewpoint: apparent shoulder width / torso length. Side-on, the shoulders nearly overlap;
  from behind or in front they are a torso-width apart. D4 suggests this ratio and says to
  calibrate the threshold on labelled clips;
- keypoint score: the pose model's median score on the body points the features use;
- a kick: a contact frame and, when contact had to come from foot speed alone, the kicking
  foot clearly moving more than the planted one there. When the ball is seen leaving, the
  ball shows the kick (in super slow motion, c20, both feet barely move near contact).
Ball visibility and frame timing don't refuse a clip. They withhold the features that need
them (plant-foot offset; velocities) and are listed as warnings.
"""

from __future__ import annotations

import numpy as np

from kickcoach.events import Events
from kickcoach.events.kick import point
from kickcoach.pose import skeleton

# heuristic, set on the tuning split's kick windows (JOURNAL, Phase 3): the side-on clips
# reached at most 0.437 (c06), the views Aamir marked not OK at least 0.564 (c23; rear views
# c26-c28: 0.64-0.70). 0.5 is about midway. All four negatives are tuning clips, so E8's
# held-out numbers can only show false refusals.
MAX_VIEW_RATIO = 0.5
# heuristic: RTMPose's median score over all clips' leg points was 0.74, 1st percentile 0.35
# (E1 data); below 0.4 for the whole kick window means the body is barely visible.
MIN_KEYPOINT_SCORE = 0.4
# heuristic: at contact the kicking foot moved 2.8-140x faster than the plant foot on the
# labelled clips (contrast in kickcoach.events.kick.leg_at_contact).
MIN_FOOT_SPEED_CONTRAST = 1.5
CORE = ("shoulder", "hip", "knee", "ankle")


def kick_window(ev: Events, n_frames: int) -> tuple[int, int] | None:
    if ev.contact_frame is None:
        return None
    c = ev.contact_frame
    lo = ev.backswing.window_start if ev.backswing else max(0, c - 10)
    return lo, min(c, n_frames - 1)


def view_ratio(xy: np.ndarray, lo: int, hi: int) -> float:
    """Median over [lo, hi] of shoulder width / torso length (0 = perfectly side-on)."""
    ls, rs = point(xy, "left_shoulder")[lo : hi + 1], point(xy, "right_shoulder")[lo : hi + 1]
    lh, rh = point(xy, "left_hip")[lo : hi + 1], point(xy, "right_hip")[lo : hi + 1]
    torso = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2, axis=-1)
    r = np.linalg.norm(ls - rs, axis=-1) / np.where(torso > 0, torso, np.nan)
    return float(np.nanmedian(r)) if np.isfinite(r).any() else float("nan")


def assess(cleaned, ev: Events, meta: dict) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []
    xy, conf = cleaned.xy, cleaned.conf
    win = kick_window(ev, len(xy))
    ratio = score = float("nan")
    if ev.kicking_leg is None or win is None:
        reasons.append("no kick found: no foot track or no contact frame")
    else:
        lo, hi = win
        ratio = view_ratio(xy, lo, hi)
        idx = [skeleton.INDEX[f"{s}_{j}"] for s in ("left", "right") for j in CORE]
        s = conf[lo : hi + 1][:, idx]
        score = float(np.nanmedian(s)) if np.isfinite(s).any() else float("nan")
        if not ratio <= MAX_VIEW_RATIO:
            reasons.append(
                f"camera not side-on: shoulder width is {ratio:.2f} of torso length in the kick "
                f"(> {MAX_VIEW_RATIO}, heuristic). Film from the side, level with the hips"
            )
        if not score >= MIN_KEYPOINT_SCORE:
            reasons.append(
                f"body points unclear: median keypoint score {score:.2f} (< {MIN_KEYPOINT_SCORE}, "
                "heuristic). Film in better light, with the whole body in frame"
            )
        if (
            ev.contact_method == "foot_speed"
            and not ev.kicking_leg.contrast >= MIN_FOOT_SPEED_CONTRAST
        ):
            reasons.append(
                f"no clear kick: at contact the kicking foot moves only "
                f"{ev.kicking_leg.contrast:.1f}x as much as the other "
                f"(< {MIN_FOOT_SPEED_CONTRAST}, heuristic)"
            )
    if ev.contact_method == "foot_speed":
        warnings.append(
            "ball not tracked: contact from peak foot speed; plant-foot offset withheld"
        )
    if not meta.get("fps_trusted"):
        warnings.append(f"velocities withheld: {meta.get('fps_reason', 'frame timing unknown')}")
    spf = meta.get("seconds_per_frame")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "viewpoint_ratio": None if np.isnan(ratio) else round(ratio, 3),
        "viewpoint_score": None if np.isnan(ratio) else round(max(0.0, 1.0 - ratio), 3),
        "keypoint_score": None if np.isnan(score) else round(score, 3),
        "capture_fps": round(1 / spf, 2) if meta.get("fps_trusted") and spf else None,
        "fps_trusted": bool(meta.get("fps_trusted")),
        "fps_reason": meta.get("fps_reason"),
    }
