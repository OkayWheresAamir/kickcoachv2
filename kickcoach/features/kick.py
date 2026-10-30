"""The Phase 1 features (handoff 02_MVP_SPEC), from cleaned keypoints filtered with a break at
contact, plus the events.

Each feature is ``measured`` with a value, or ``not_measured`` with the reason: a missing
event or point, a sanity bound (kickcoach.features.bounds), or frame timing that can't
support a velocity (DEC-010). Nothing is clamped.

``ankle_angle_contact`` is dropped (Aamir, 2026-10-30): E1 put only 16% of RTMPose's
kicking-toe points at contact within 0.1 thigh lengths, and the angle came out 28 deg too
small on average (results/e1_summary.md).

``keypoint_score`` is the pose model's mean score for the points a feature uses: the model's
own number (RTMPose's SimCC peak), not a calibrated probability (DEC-011).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kickcoach.events import Events
from kickcoach.events.kick import knee_angle, other, point
from kickcoach.features.bounds import violation
from kickcoach.features.geometry import lean_from_vertical_deg
from kickcoach.pose import skeleton
from kickcoach.signal import derivative

# Handoff D6: velocities need a trusted capture rate of at least 120 fps (configurable).
MIN_VELOCITY_FPS = 120.0


@dataclass
class Feature:
    id: str
    value: float | str | None
    unit: str
    status: str  # "measured" | "not_measured"
    reason: str | None = None
    frame: int | None = None
    keypoint_score: float | None = None
    extra: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _missing(fid: str, unit: str, reason: str) -> Feature:
    return Feature(fid, None, unit, "not_measured", reason)


def _score(conf: np.ndarray, t: int, names: tuple[str, ...]) -> float | None:
    s = conf[t, [skeleton.INDEX[n] for n in names]]
    return round(float(np.nanmean(s)), 3) if np.isfinite(s).any() else None


def _bounded(fid: str, value: float, unit: str, **kw) -> Feature:
    if not np.isfinite(value):
        return _missing(fid, unit, "a needed keypoint is missing at that frame")
    why = violation(fid, value)
    if why:
        return Feature(fid, None, unit, "not_measured", why, **kw)
    return Feature(fid, round(float(value), 2), unit, "measured", **kw)


def segment_angle(xy: np.ndarray, upper: str, lower: str, facing: int) -> np.ndarray:
    """Angle of upper->lower from downward vertical, + toward the kicking direction, in
    unwrapped degrees (so a forward swing increases it)."""
    d = point(xy, lower) - point(xy, upper)
    return np.degrees(np.unwrap(np.arctan2(facing * d[:, 0], d[:, 1])))


def compute(
    xy: np.ndarray,
    conf: np.ndarray,
    ev: Events,
    *,
    seconds_per_frame: float | None,
    fps_trusted: bool,
    evenly_timed: bool,
) -> list[Feature]:
    """``xy`` must be filtered with a break at ``ev.contact_frame`` (see kickcoach.pipeline).
    ``evenly_timed`` = constant frame rate with no repeated frames, so frame order and
    per-frame rates mean something even when the real time per frame is unknown."""
    c, kl, facing = ev.contact_frame, ev.kicking_leg, ev.facing
    if kl is None or c is None:
        reason = "no kick found (kicking leg or contact frame not detected)"
        return [_missing(f, u, reason) for f, u in FEATURE_UNITS.items()]
    k, p = kl.leg, other(kl.leg)
    knee = knee_angle(xy, k)
    leg_pts = (f"{k}_hip", f"{k}_knee", f"{k}_ankle")
    out = [
        _bounded(
            "knee_angle_contact", knee[c], "deg", frame=c, keypoint_score=_score(conf, c, leg_pts)
        )
    ]

    bs = ev.backswing
    if bs is None:
        out.append(
            _missing("max_knee_flexion_backswing", "deg", "no backswing found before contact")
        )
    else:
        out.append(
            _bounded(
                "max_knee_flexion_backswing", knee[bs.frame], "deg", frame=bs.frame,
                keypoint_score=_score(conf, bs.frame, leg_pts),
                extra={"window_from_clip_start": True} if bs.window_from_clip_start else None,
            )
        )  # fmt: skip

    trunk_pts = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if facing is None:
        out.append(
            _missing("trunk_lean_contact", "deg", "kick direction unknown, so lean has no sign")
        )
    else:
        hip_mid = (point(xy, "left_hip") + point(xy, "right_hip")) / 2
        sh_mid = (point(xy, "left_shoulder") + point(xy, "right_shoulder")) / 2
        lean = lean_from_vertical_deg(hip_mid[c], sh_mid[c], facing)
        out.append(
            _bounded(
                "trunk_lean_contact",
                lean,
                "deg",
                frame=c,
                keypoint_score=_score(conf, c, trunk_pts),
            )
        )

    rest = ev.ball_rest
    if rest is None or ev.contact_method == "foot_speed":
        out.append(_missing("plant_foot_offset", "leg_lengths", "ball not tracked at contact"))
    elif facing is None:
        out.append(_missing("plant_foot_offset", "leg_lengths", "kick direction unknown"))
    else:
        hip, kn, an = (point(xy, f"{p}_{j}")[c] for j in ("hip", "knee", "ankle"))
        leg_len = np.linalg.norm(hip - kn) + np.linalg.norm(kn - an)
        off = facing * (an[0] - rest.x) / leg_len if leg_len > 0 else np.nan
        out.append(
            _bounded(
                "plant_foot_offset", off, "leg_lengths", frame=c,
                keypoint_score=_score(conf, c, (f"{p}_hip", f"{p}_knee", f"{p}_ankle")),
            )
        )  # fmt: skip

    out.append(_sequence(xy, ev, seconds_per_frame, fps_trusted, evenly_timed))
    out.append(_knee_velocity(knee, ev, seconds_per_frame, fps_trusted))
    return out


def _sequence(xy, ev: Events, spf, fps_trusted: bool, evenly_timed: bool) -> Feature:
    fid = "segment_sequence_order"
    if not evenly_timed:
        return _missing(fid, "order", "frames are not evenly spaced in time")
    if ev.facing is None or ev.backswing is None:
        return _missing(fid, "order", "kick direction or backswing not found")
    k, c = ev.kicking_leg.leg, ev.contact_frame
    lo = ev.backswing.window_start
    breaks = (c,)
    thigh_w = derivative(segment_angle(xy, f"{k}_hip", f"{k}_knee", ev.facing), breaks=breaks)
    shank_w = derivative(segment_angle(xy, f"{k}_knee", f"{k}_ankle", ev.facing), breaks=breaks)
    tw, sw = thigh_w[lo : c + 1], shank_w[lo : c + 1]
    if not (np.isfinite(tw).any() and np.isfinite(sw).any()):
        return _missing(fid, "order", "thigh or shank track missing in the forward swing")
    t_peak, s_peak = lo + int(np.nanargmax(tw)), lo + int(np.nanargmax(sw))
    order = (
        "proximal_to_distal" if t_peak < s_peak
        else "distal_to_proximal" if t_peak > s_peak
        else "simultaneous"
    )  # fmt: skip
    extra = {"thigh_peak_frame": t_peak, "shank_peak_frame": s_peak}
    # Handoff D6: timing in ms, like velocities, needs a trusted rate of at least 120 fps.
    if fps_trusted and spf and 1.0 / spf >= MIN_VELOCITY_FPS:
        extra["lag_ms"] = round((s_peak - t_peak) * spf * 1000, 1)
        shank_peak = float(np.nanmax(sw)) / spf
        why = violation("peak_shank_ang_velocity", shank_peak)
        if why:
            return Feature(fid, None, "order", "not_measured", f"shank angular velocity: {why}")
        extra["peak_shank_ang_velocity_deg_s"] = round(shank_peak, 1)
    else:
        extra["lag_ms"] = None
        extra["lag_note"] = (
            f"lag in ms withheld: capture rate {1 / spf:.0f} fps is below "
            f"{MIN_VELOCITY_FPS:.0f} (handoff D6)"
            if fps_trusted and spf
            else "lag in ms withheld: real time per frame unknown (DEC-010)"
        )
    return Feature(fid, order, "order", "measured", frame=c, extra=extra)


def _knee_velocity(knee: np.ndarray, ev: Events, spf, fps_trusted: bool) -> Feature:
    fid = "peak_knee_ext_velocity"
    if not (fps_trusted and spf):
        return _missing(
            fid, "deg/s", "capture frame rate unknown, so no real time per frame (DEC-010)"
        )
    if 1.0 / spf < MIN_VELOCITY_FPS:
        return _missing(
            fid,
            "deg/s",
            f"capture rate {1 / spf:.0f} fps is below {MIN_VELOCITY_FPS:.0f} (handoff D6)",
        )
    if ev.backswing is None:
        return _missing(fid, "deg/s", "no backswing found before contact")
    c, b = ev.contact_frame, ev.backswing.frame
    w = derivative(knee, dt=spf, breaks=(c,))[b : c + 1]
    if not np.isfinite(w).any():
        return _missing(fid, "deg/s", "knee track missing between backswing and contact")
    t = b + int(np.nanargmax(w))
    return _bounded(fid, float(np.nanmax(w)), "deg/s", frame=t)


FEATURE_UNITS = {
    "knee_angle_contact": "deg",
    "max_knee_flexion_backswing": "deg",
    "trunk_lean_contact": "deg",
    "plant_foot_offset": "leg_lengths",
    "segment_sequence_order": "order",
    "peak_knee_ext_velocity": "deg/s",
}
