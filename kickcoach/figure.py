"""The stick figure the app draws for an accepted clip: the cleaned keypoints at contact and at
the deepest backswing.

The upload is deleted after analysis, so this is the only picture of the kick that is kept.
It holds no pixels: just the filtered positions the angles were measured on, in source-frame
pixels (y down), rounded to 0.1 px.
"""

from __future__ import annotations

import numpy as np

from kickcoach.pose.skeleton import KEYPOINTS

FRAMES = {"contact": "contact_frame", "max_backswing": "max_backswing_frame"}


def frame_size(meta: dict) -> tuple[int, int]:
    """Upright (width, height). Ingests before rotation handling have no frame_size, and
    none of those clips was rotated."""
    if "frame_size" in meta:
        w, h = meta["frame_size"]
        return int(w), int(h)
    return int(meta["stream"]["width"]), int(meta["stream"]["height"])


def pose_figure(xy: np.ndarray, result: dict, size: tuple[int, int]) -> dict | None:
    """``xy`` is the cleaned [T, K, 2] signal ``result`` was measured on. None for a refused
    clip: drawing a skeleton there would suggest a measurement the gate declined to make."""
    if not result["quality"]["accepted"]:
        return None
    ev = result["events"]
    frames = {}
    for name, key in FRAMES.items():
        t = ev.get(key)
        if t is None:
            continue
        frames[name] = {
            "frame": int(t),
            "points": {
                joint: [round(float(x), 1), round(float(y), 1)]
                if np.isfinite(x) and np.isfinite(y)
                else None
                for joint, (x, y) in zip(KEYPOINTS, xy[t], strict=True)
            },
        }
    if not frames:
        return None
    return {
        "frame_size": list(size),
        "kicking_leg": ev.get("kicking_leg"),
        "kick_direction": ev.get("kick_direction"),
        "ball_rest": ev.get("ball_rest"),
        "frames": frames,
    }
