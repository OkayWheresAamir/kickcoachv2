"""The measurement pipeline for one clip: pose -> signal -> events -> quality -> features.

Two cleaning passes: events are found on a signal filtered straight through, then the
signal is re-filtered with a break at the contact frame, so impact doesn't smear into the
swing (Nunome et al. 2006, p. 12), and the backswing and features come from that.
Output follows the handoff's MVP response shape (02_MVP_SPEC.md), without the parts later
phases add (comparisons, faults, feedback, history).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kickcoach import events
from kickcoach.events.kick import knee_angle, max_backswing
from kickcoach.features import kick as features
from kickcoach.ingest.fps import MAX_REPEATED_FRAME_RATIO
from kickcoach.quality import gate
from kickcoach.signal import clean


def analyse(
    poses: np.ndarray,
    meta: dict,
    ball_cands: list[np.ndarray] | None,
    *,
    kicking_leg: str | None = None,
    kick_direction: str | None = None,
    target_cpf: float | None | str = "auto",
) -> dict:
    """``meta`` is the ingest metadata (kickcoach.ingest): fps_trusted, seconds_per_frame,
    constant_frame_rate, repeated_frame_ratio. ``kicking_leg``/``kick_direction`` are
    optional user inputs ("left"/"right"); otherwise they are detected. ``target_cpf``
    overrides the cutoff, for E2's sensitivity check only."""
    spf = meta["seconds_per_frame"] if meta.get("fps_trusted") else None
    first = clean(poses, seconds_per_frame=spf, target_cpf=target_cpf)
    ev = events.detect(
        first.xy, ball_cands, kicking_leg_hint=kicking_leg, kick_direction_hint=kick_direction
    )
    cleaned = first
    if ev.contact_frame is not None:
        cleaned = clean(
            poses, seconds_per_frame=spf, breaks=(ev.contact_frame,), target_cpf=target_cpf
        )
        ev = replace(
            ev,
            backswing=max_backswing(knee_angle(cleaned.xy, ev.kicking_leg.leg), ev.contact_frame),
        )
    q = gate.assess(cleaned, ev, meta)
    evenly_timed = bool(meta.get("constant_frame_rate")) and (
        meta.get("repeated_frame_ratio", 1.0) <= MAX_REPEATED_FRAME_RATIO
    )
    feats = (
        features.compute(
            cleaned.xy, cleaned.conf, ev,
            seconds_per_frame=spf, fps_trusted=bool(meta.get("fps_trusted")),
            evenly_timed=evenly_timed,
        )
        if q["accepted"]
        else []
    )  # fmt: skip
    return {
        "quality": q,
        "events": events_dict(ev),
        "features": [f.to_dict() for f in feats],
        "signal": {
            "cutoff_cpf": cleaned.cutoff_cpf,
            "breaks": list(cleaned.breaks),
            "swapped_frames": [int(t) for t in np.flatnonzero(cleaned.swapped)],
            "glitch_frames": [int(t) for t in np.flatnonzero(cleaned.glitches.any(1))],
            "filled_points": int(cleaned.filled.sum()),
        },
    }


def events_dict(ev: events.Events) -> dict:
    kl, bs, rest = ev.kicking_leg, ev.backswing, ev.ball_rest
    return {
        "kicking_leg": kl.leg if kl else None,
        "kicking_leg_source": kl.source if kl else None,
        "foot_speed_contrast": round(kl.contrast, 2) if kl else None,
        "kick_direction": None if ev.facing is None else ("right" if ev.facing == 1 else "left"),
        "kick_direction_source": ev.facing_source,
        "contact_frame": ev.contact_frame,
        "contact_method": ev.contact_method,
        "contact_confidence": ev.contact_confidence,
        "max_backswing_frame": bs.frame if bs else None,
        "backswing_window_start": bs.window_start if bs else None,
        "ball_rest": None
        if rest is None
        else {"x": round(rest.x, 1), "y": round(rest.y, 1), "diameter": round(rest.diameter, 1)},
        "ball_last_at_rest": ev.ball_contact.frame if ev.ball_contact else None,
        "ball_next_seen_moved": ev.ball_contact.next_seen if ev.ball_contact else None,
        "notes": list(ev.notes),
    }
