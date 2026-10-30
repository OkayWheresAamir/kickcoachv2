"""Events: kicking leg, kick direction, ball track, contact frame, max backswing (handoff D8,
D9). Run on cleaned keypoints (kickcoach.signal) without a break; features then re-clean with
a break at the contact found here.

The ball comes first, because finding contact from it doesn't need the kicking leg; the
kicking leg is then the foot moving into the ball. Pose alone can't reliably say when the
kick happens: a running stride also has one foot swinging and the other planted.

Contact, in order of preference:
1. ``ball_motion``: the ball is seen at rest, then seen moved on the very next frame.
2. ``ball_bracket``: the ball is last seen at rest on frame L and next seen moved on frame N
   (or never). The touch is in [L, N - 1], hidden by the foot or the body; the frame where
   one foot is fastest relative to the other inside that bracket picks it. On the tuning
   clips this ran 0-3 frames early (median -2): the foot peaks just before it is pressed
   into the ball. Left uncorrected; E3 (Phase 4) decides on the held-out clips.
3. ``foot_speed``: no resting ball found: where one foot is fastest relative to the other
   (the handoff's fallback, D9; unreliable, see kick.leg_from_speed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kickcoach.events.ball import BallContact, Rest, contact_from_ball, rest_position
from kickcoach.events.kick import (
    Backswing,
    KickingLeg,
    kick_direction,
    kick_in,
    knee_angle,
    leg_at_contact,
    leg_from_speed,
    max_backswing,
    point,
)
from kickcoach.signal.swaps import thigh_scale

# heuristic: at contact the kicking ankle must be within this many thigh lengths of the
# resting ball, or the ball-based contact is rejected (it was probably another round thing).
REACH_THIGHS = 1.5


@dataclass(frozen=True)
class Events:
    kicking_leg: KickingLeg | None
    facing: int | None  # +1 = kicks toward image right
    facing_source: str
    contact_frame: int | None
    contact_method: str | None  # "ball_motion" | "ball_bracket" | "foot_speed"
    contact_confidence: str | None  # "high" | "medium" | "low"
    ball_rest: Rest | None
    ball_contact: BallContact | None
    backswing: Backswing | None
    notes: tuple[str, ...]


def detect(
    xy: np.ndarray,
    ball_cands: list[np.ndarray] | None,
    *,
    kicking_leg_hint: str | None = None,
    kick_direction_hint: str | None = None,
) -> Events:
    notes: list[str] = []
    scale = thigh_scale(xy)
    rest = rest_position(ball_cands) if ball_cands else None
    bc = contact_from_ball(ball_cands, rest) if rest else None
    if ball_cands is not None and rest is None:
        notes.append("no resting ball found")

    kl = contact = method = conf = None
    if bc is not None:
        if bc.confidence == "high":
            contact, method, conf = bc.frame, "ball_motion", "high"
        else:
            hi = (bc.next_seen if bc.next_seen is not None else len(xy)) - 1
            contact = kick_in(xy, bc.frame, hi, kicking_leg_hint)
            method, conf = "ball_bracket", bc.confidence
        kl = leg_at_contact(xy, contact, kicking_leg_hint) if contact is not None else None
        reach = (
            np.linalg.norm(point(xy, f"{kl.leg}_ankle")[contact] - [rest.x, rest.y]) / scale
            if kl is not None
            else np.nan
        )
        if not reach <= REACH_THIGHS:
            notes.append(
                f"ball-based contact rejected: kicking ankle {reach:.1f} thigh lengths from the "
                f"ball at contact (> {REACH_THIGHS}, heuristic)"
            )
            kl = contact = method = conf = None
    if contact is None:
        kl = leg_from_speed(xy, kicking_leg_hint)
        if kl is None:
            return Events(
                None, None, "none", None, None, None, rest, bc, None, (*notes, "no foot track")
            )
        contact, method, conf = kl.kick_frame, "foot_speed", "low"

    if kick_direction_hint in ("left", "right"):
        facing, facing_source = (1 if kick_direction_hint == "right" else -1), "given"
    else:
        facing, facing_source = kick_direction(xy, kl.leg, contact), "detected"
    bs = max_backswing(knee_angle(xy, kl.leg), contact)
    return Events(kl, facing, facing_source, contact, method, conf, rest, bc, bs, tuple(notes))
