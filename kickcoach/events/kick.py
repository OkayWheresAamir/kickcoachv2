"""Kick events from cleaned keypoints: kicking leg, kick direction, contact by peak foot speed
(the fallback when the ball can't be tracked, handoff D9), and the max-backswing frame.

Foot speed uses the ankle: E1 placed it best of the foot points (91% PCK@0.1 for RTMPose, vs
71% heel and 63% big toe). Speeds are in thigh lengths per frame, so they are comparable
across resolutions; they are event cues only, never features (DEC-010).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kickcoach.features.geometry import joint_angle_deg
from kickcoach.pose import skeleton
from kickcoach.signal import derivative
from kickcoach.signal.swaps import thigh_scale

SIDES = ("left", "right")
FOOT = "ankle"
# heuristic: the backswing valley must dip at least this far below the contact angle, so a
# wobble just before contact isn't mistaken for the whole backswing.
MIN_VALLEY_DEPTH_DEG = 20.0


def point(xy: np.ndarray, name: str) -> np.ndarray:
    return xy[:, skeleton.INDEX[name]]


def other(leg: str) -> str:
    return "left" if leg == "right" else "right"


def foot_speed(xy: np.ndarray, leg: str, scale: float, breaks: tuple[int, ...] = ()) -> np.ndarray:
    """[T] speed of the leg's ankle, thigh lengths per frame."""
    v = derivative(point(xy, f"{leg}_{FOOT}"), breaks=breaks)
    return np.linalg.norm(v, axis=-1) / scale


def knee_angle(xy: np.ndarray, leg: str) -> np.ndarray:
    return joint_angle_deg(
        point(xy, f"{leg}_hip"), point(xy, f"{leg}_knee"), point(xy, f"{leg}_ankle")
    )


# heuristic: the kicking foot is the one that has moved this many thigh lengths (net) by
# contact, looking back only as far as needed. A net displacement doesn't depend on how slow
# the clip's motion is, and jitter doesn't accumulate in it as it would in a path length.
MIN_SWING_THIGHS = 0.5
# Frames up to contact used for the kick direction (heuristic: inside the kick at 30 fps).
CONTACT_WINDOW = 2


@dataclass(frozen=True)
class KickingLeg:
    leg: str
    kick_frame: int  # the frame the decision was made at (contact, or the speed peak)
    contrast: float  # kicking foot motion / other foot motion there (higher = clearer)
    source: str  # "given" | "ball_window" | "speed_difference"


def _speeds(xy: np.ndarray, lo: int = 0, hi: int | None = None) -> dict[str, np.ndarray]:
    scale = thigh_scale(xy)
    return {leg: foot_speed(xy, leg, scale)[lo:hi] for leg in SIDES}


def leg_at_contact(xy: np.ndarray, contact: int, given: str | None = None) -> KickingLeg | None:
    """The foot that swung into the ball: the larger net displacement up to ``contact``, over
    the shortest look-back in which one foot has moved MIN_SWING_THIGHS. The plant foot is
    planted then, so it has hardly moved.

    Found from the ball, contact doesn't depend on which leg kicks, so it can pick the leg.
    Speeds over a fixed 2 frames failed on c20 (super slow motion): both feet barely moved.
    """
    scale = thigh_scale(xy)
    ank = {leg: point(xy, f"{leg}_{FOOT}") for leg in SIDES}
    if not all(np.isfinite(a[contact]).all() for a in ank.values()):
        return None
    disp = {leg: 0.0 for leg in SIDES}
    for k in range(1, contact + 1):
        d = {leg: np.linalg.norm(a[contact] - a[contact - k]) / scale for leg, a in ank.items()}
        if not all(np.isfinite(v) for v in d.values()):
            continue
        disp = d
        if max(d.values()) >= MIN_SWING_THIGHS:
            break
    if given in SIDES:
        leg, source = given, "given"
    else:
        leg, source = max(SIDES, key=lambda s: disp[s]), "ball_window"
    o = disp[other(leg)]
    return KickingLeg(leg, contact, float(disp[leg] / o) if o > 0 else np.inf, source)


def kick_in(xy: np.ndarray, lo: int, hi: int, given: str | None = None) -> int | None:
    """Frame in [lo, hi] where one foot is fastest relative to the other (or ``given``'s)."""
    sp = _speeds(xy, lo, hi + 1)
    legs = [given] if given in SIDES else list(SIDES)
    best = None
    for leg in legs:
        d = sp[leg] - sp[other(leg)]
        if np.isfinite(d).any():
            t = int(np.nanargmax(d))
            if best is None or d[t] > best[1]:
                best = (lo + t, float(d[t]))
    return None if best is None else best[0]


def leg_from_speed(xy: np.ndarray, given: str | None = None) -> KickingLeg | None:
    """Without a ball: the frame where one foot is fastest relative to the other.

    Unreliable, and reported as such: a running stride also has one foot swinging and the
    other planted, and on the labelled clips this peak landed 1-68 frames from contact
    (JOURNAL, Phase 3). It is the fallback the handoff names (D9), not a good detector.
    """
    sp = _speeds(xy)
    cands = [given] if given in SIDES else list(SIDES)
    best = None
    for leg in cands:
        d = sp[leg] - sp[other(leg)]
        if np.isfinite(d).any():
            t = int(np.nanargmax(d))
            if best is None or d[t] > best[2]:
                best = (leg, t, float(d[t]))
    if best is None:
        return None
    leg, t, _ = best
    o = sp[other(leg)][t]
    return KickingLeg(leg, t, float(sp[leg][t] / o) if o > 0 else np.inf,
                      "given" if given in SIDES else "speed_difference")  # fmt: skip


def kick_direction(xy: np.ndarray, leg: str, frame: int) -> int | None:
    """+1 if the kicking ankle moves toward image right over the frames up to ``frame``, -1
    toward image left. The trunk-lean sign depends on it (DEC-006)."""
    vx = derivative(point(xy, f"{leg}_{FOOT}")[:, 0])[max(0, frame - CONTACT_WINDOW) : frame + 1]
    if not np.isfinite(vx).any() or np.nanmean(vx) == 0:
        return None
    return 1 if np.nanmean(vx) > 0 else -1


@dataclass(frozen=True)
class Backswing:
    frame: int  # deepest knee flexion in the valley before contact
    window_start: int
    angle_deg: float
    window_from_clip_start: bool  # no valley edge found: the search ran to the clip start


def max_backswing(knee: np.ndarray, contact: int) -> Backswing | None:
    """Deepest knee flexion in the flexion valley that ends at contact.

    Walking back from contact, the knee angle falls into the backswing valley and then rises
    again (the stride before the kick). The valley's edge is the first frame back at least as
    straight as at contact, once the valley has dipped MIN_VALLEY_DEPTH_DEG below it. Stopping
    there keeps an earlier stride's knee bend out of the search. Ties go to the earlier frame,
    as in the labelling guide. Our convention: the literature defines the frame (Kellis & Katis
    2007, p. 155), not how to bound the search.
    """
    ok = np.flatnonzero(np.isfinite(knee[: contact + 1]))
    if len(ok) == 0:
        return None
    ref_t = ok[-1]  # contact, or the last valid frame before it
    ref = knee[ref_t]
    start, from_clip_start = int(ok[0]), True
    run_min = np.inf
    for t in ok[::-1]:
        run_min = min(run_min, knee[t])
        if knee[t] >= ref and run_min < ref - MIN_VALLEY_DEPTH_DEG:
            start, from_clip_start = int(t), False
            break
    seg = knee[start : contact + 1]
    if not np.isfinite(seg).any():
        return None
    f = start + int(np.nanargmin(seg))
    return Backswing(f, start, float(knee[f]), from_clip_start)
