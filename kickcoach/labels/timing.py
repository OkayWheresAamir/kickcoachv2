"""Labelling time, as logged by the tool, and the projection used to set the labelling budget.

active_s counts only seconds with mouse or keyboard input in the last IDLE_S seconds
(tools/label_ui/app.js); wall_s counts every second the frame was on screen.
"""

from __future__ import annotations

import statistics

from kickcoach.labels.schema import frame_roles


def frame_rows(labels: dict[str, dict]) -> list[dict]:
    rows = []
    for cid, lab in sorted(labels.items()):
        roles = frame_roles(lab["clip"], lab["n_frames"])
        for key in sorted(lab["frames"], key=int):
            e = lab["frames"][key]
            rows.append(
                {
                    "clip_id": cid,
                    "frame": int(key),
                    "roles": "+".join(roles.get(int(key), [])) or "extra",
                    "confirmed": bool(e.get("confirmed")),
                    "active_s": round(e.get("active_s", 0.0), 1),
                    "wall_s": round(e.get("wall_s", 0.0), 1),
                }
            )
    return rows


def clip_rows(labels: dict[str, dict]) -> list[dict]:
    rows = []
    for cid, lab in sorted(labels.items()):
        t = lab.get("clip_timing", {})
        frames = [e for e in lab["frames"].values() if e.get("confirmed")]
        rows.append(
            {
                "clip_id": cid,
                "purpose": lab.get("purpose"),
                "events_active_s": round(t.get("active_s", 0.0), 1),
                "events_wall_s": round(t.get("wall_s", 0.0), 1),
                "frames_confirmed": len(frames),
                "frames_active_s": round(sum(e.get("active_s", 0.0) for e in frames), 1),
            }
        )
    return rows


def projection_minutes(
    events_s: float, frame_s: float, n_event_clips: int, n_kp_clips: int, frames_per_clip: int
) -> float:
    """Events (leg, direction, view, contact, backswing) on n_event_clips, plus keypoints on
    frames_per_clip frames of n_kp_clips of them."""
    return (n_event_clips * events_s + n_kp_clips * frames_per_clip * frame_s) / 60


def summary(labels: dict[str, dict]) -> dict | None:
    """Median active seconds per confirmed frame and per clip's events, over finished clips."""
    frames = [r["active_s"] for r in frame_rows(labels) if r["confirmed"]]
    events = [
        r["events_active_s"]
        for r in clip_rows(labels)
        if r["purpose"] == "eval" and r["events_active_s"] > 0
    ]
    if not frames or not events:
        return None
    return {
        "n_frames": len(frames),
        "n_clips": len(events),
        "frame_s_median": statistics.median(frames),
        "frame_s_max": max(frames),
        "events_s_median": statistics.median(events),
        "events_s_max": max(events),
    }
