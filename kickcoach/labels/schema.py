"""Label files: data/labels/<clip_id>.json, plus data/labels/repeat/<session>.json for the
blind relabel. Definitions of every field are in docs/LABELLING_GUIDE.md.

Coordinates are continuous source-frame pixels, (0, 0) at the image's top-left corner, so
pixel (i, j) covers [i, i+1) x [j, j+1); y points down (DEC-006). Frame index = JPEG index
in data/frames/<clip>/ = pose-array row.

Each labelled frame keeps the model pre-fill it started from ("prefill"). That is model
output, not ground truth; it is stored so E1 can measure how far labels moved from it
(pre-fill anchoring, DEC-013).
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from kickcoach.pose import skeleton

LABELS_DIR = Path(__file__).resolve().parents[2] / "data" / "labels"
SCHEMA_VERSION = 1
PREFILL_MODEL = "rtmpose"

# What E1-E3 need (eval plan, "Ground truth"): shoulders, hips, knees, ankles, and the heel
# and big toe of each foot. Small toes are not labelled: from the side, a shoe hides which
# toe is which (DEC-013).
POINTS: tuple[str, ...] = (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_big_toe", "right_big_toe",
)  # fmt: skip
BALL = "ball"
ALL_POINTS = (*POINTS, BALL)

# COCO's visibility flags (cocodataset.org, "Data format"): 0 = not labelled,
# 1 = labelled but not visible, 2 = labelled and visible.
UNKNOWN, OCCLUDED, VISIBLE = 0, 1, 2

SIDES = ("left", "right")
YES_NO = ("y", "n")
ROLES = ("max_backswing", "pre_contact", "contact")
# The labelling budget (DEC-014) needs keypoints on these; contact-1 is optional.
REQUIRED_ROLES = ("max_backswing", "contact")


def path(clip_id: str, root: Path = LABELS_DIR) -> Path:
    return root / f"{clip_id}.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_label(clip_id: str, n_frames: int, width: int, height: int, purpose: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "purpose": purpose,
        "n_frames": n_frames,
        "image_size": [width, height],
        "prefill_model": PREFILL_MODEL,
        "clip": {
            "kicking_leg": None,
            "kick_direction": None,
            "view_ok_human": None,
            "contact_frame": None,
            "max_backswing_frame": None,
            "contact_between_frames": False,
            "notes": "",
        },
        "clip_timing": {"active_s": 0.0, "wall_s": 0.0},
        "frames": {},
    }


def prefill_points(pose_row: np.ndarray) -> dict[str, list[float] | None]:
    """One canonical [K, 3] pose row -> {point: [x, y, conf] or None} for the labelled points."""
    out: dict[str, list[float] | None] = {}
    for name in POINTS:
        x, y, c = (float(v) for v in pose_row[skeleton.INDEX[name]])
        out[name] = None if math.isnan(x) or math.isnan(y) else [x, y, c]
    out[BALL] = None  # no ball detector yet (Phase 3)
    return out


def frame_roles(clip: dict, n_frames: int) -> dict[int, list[str]]:
    """Frames that need keypoints, from the clip's events: backswing, contact-1, contact."""
    roles: dict[int, list[str]] = {}
    b, c = clip.get("max_backswing_frame"), clip.get("contact_frame")
    for idx, role in (
        (b, "max_backswing"),
        (c - 1 if c is not None else None, "pre_contact"),
        (c, "contact"),
    ):
        if idx is not None and 0 <= idx < n_frames:
            roles.setdefault(idx, []).append(role)
    return roles


def required_frames(clip: dict, n_frames: int) -> list[int]:
    roles = frame_roles(clip, n_frames)
    return sorted(i for i, rs in roles.items() if set(rs) & set(REQUIRED_ROLES))


def points_array(points: dict) -> np.ndarray:
    """{name: {x, y, v}} -> [len(ALL_POINTS), 3] of (x, y, v); unknown points are (nan, nan, 0)."""
    out = np.full((len(ALL_POINTS), 3), np.nan)
    out[:, 2] = UNKNOWN
    for i, name in enumerate(ALL_POINTS):
        p = points.get(name)
        if p and p.get("v", UNKNOWN) > UNKNOWN:
            out[i] = (p["x"], p["y"], p["v"])
    return out


def validate(label: dict) -> list[str]:
    """Structural errors in a clip label file (empty list = valid). Judgement calls, like a
    backswing after contact, are QA flags (kickcoach.labels.qa), not errors."""
    errs: list[str] = []
    if label.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    n = label.get("n_frames")
    size = label.get("image_size")
    if not isinstance(n, int) or n <= 0:
        return [*errs, "n_frames must be a positive int"]
    if not (isinstance(size, list) and len(size) == 2):
        return [*errs, "image_size must be [width, height]"]
    clip = label.get("clip", {})
    for key, allowed in (
        ("kicking_leg", SIDES),
        ("kick_direction", SIDES),
        ("view_ok_human", YES_NO),
    ):
        if clip.get(key) not in (None, *allowed):
            errs.append(f"clip.{key} must be one of {allowed} or null")
    for key in ("contact_frame", "max_backswing_frame"):
        v = clip.get(key)
        if v is not None and not (isinstance(v, int) and 0 <= v < n):
            errs.append(f"clip.{key} must be a frame index in [0, {n})")
    for key, entry in label.get("frames", {}).items():
        if not (key.isdigit() and int(key) < n):
            errs.append(f"frames key {key!r} is not a frame index in [0, {n})")
        errs += [f"frame {key}: {e}" for e in validate_points(entry.get("points", {}), size)]
    return errs


def validate_points(points: dict, size: list) -> list[str]:
    errs = []
    w, h = size
    for name, p in points.items():
        if name not in ALL_POINTS:
            errs.append(f"unknown point {name!r}")
            continue
        v = p.get("v")
        if v not in (UNKNOWN, OCCLUDED, VISIBLE):
            errs.append(f"{name}: v must be 0, 1 or 2")
        elif v == UNKNOWN:
            if p.get("x") is not None or p.get("y") is not None:
                errs.append(f"{name}: an unknown point (v=0) has no coordinates")
        elif not all(isinstance(p.get(k), (int, float)) for k in ("x", "y")):
            errs.append(f"{name}: a labelled point needs numeric x and y")
        elif not (0 <= p["x"] <= w and 0 <= p["y"] <= h):
            errs.append(f"{name}: ({p['x']}, {p['y']}) is outside the {w}x{h} image")
    return errs


def _round_points(points: dict) -> dict:
    out = {}
    for name, p in points.items():
        q = dict(p)
        for k in ("x", "y"):
            if isinstance(q.get(k), float):
                q[k] = round(q[k], 2)
        out[name] = q
    return out


def _write_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)  # atomic: a crash mid-write never leaves a half-written label file


def save(label: dict, root: Path = LABELS_DIR) -> Path:
    errs = validate(label)
    if errs:
        raise ValueError("; ".join(errs))
    label = {
        **label,
        "frames": {
            k: {**e, "points": _round_points(e.get("points", {}))}
            for k, e in label.get("frames", {}).items()
        },
    }
    p = path(label["clip_id"], root)
    _write_json(p, label)
    return p


def load(clip_id: str, root: Path = LABELS_DIR) -> dict | None:
    p = path(clip_id, root)
    return json.loads(p.read_text()) if p.exists() else None


def load_all(root: Path = LABELS_DIR) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(root.glob("c*.json"))}


def status(label: dict) -> dict:
    """Progress for the clip list: are the events set, and how many required frames are
    confirmed."""
    clip = label["clip"]
    if label.get("purpose") == "gate_only":
        return {
            "events_done": clip["view_ok_human"] is not None,
            "frames_needed": 0,
            "frames_confirmed": 0,
            "complete": clip["view_ok_human"] is not None,
        }
    events_done = all(
        clip[k] is not None
        for k in (
            "kicking_leg",
            "kick_direction",
            "view_ok_human",
            "contact_frame",
            "max_backswing_frame",
        )
    )
    needed = required_frames(clip, label["n_frames"])
    confirmed = sum(1 for i in needed if label["frames"].get(str(i), {}).get("confirmed"))
    return {
        "events_done": events_done,
        "frames_needed": len(needed),
        "frames_confirmed": confirmed,
        "complete": events_done and confirmed == len(needed) > 0,
    }


# --- blind relabel (labelling repeatability) ---------------------------------------------


def repeat_dir(root: Path = LABELS_DIR) -> Path:
    return root / "repeat"


def sample_repeat_tasks(
    labels: dict[str, dict], n: int, seed: int, min_age_hours: float, now: datetime | None = None
) -> list[dict]:
    """Pick n confirmed frames to relabel blind, spread across clips (round-robin over a
    shuffled clip order). Frames confirmed less than min_age_hours ago are skipped, so the
    first labels are not fresh in memory."""
    now = now or datetime.now().astimezone()
    cutoff = now - timedelta(hours=min_age_hours)
    pools: dict[str, list[int]] = {}
    for cid, lab in sorted(labels.items()):
        if lab.get("purpose") != "eval":
            continue
        ok = [
            int(k)
            for k, e in lab["frames"].items()
            if e.get("confirmed") and datetime.fromisoformat(e["confirmed_at"]) <= cutoff
        ]
        if ok:
            pools[cid] = sorted(ok)
    rng = random.Random(seed)
    order = sorted(pools)
    rng.shuffle(order)
    for cid in order:
        rng.shuffle(pools[cid])
    tasks = []
    while len(tasks) < n and any(pools.values()):
        for cid in order:
            if pools[cid] and len(tasks) < n:
                tasks.append({"clip_id": cid, "frame": pools[cid].pop()})
    return tasks


def new_repeat_session(session_id: str, seed: int, tasks: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "repeatability",
        "session_id": session_id,
        "created_at": now_iso(),
        "seed": seed,
        "tasks": [
            {
                **t,
                "points": {},
                "prefill": {},
                "confirmed": False,
                "confirmed_at": None,
                "active_s": 0.0,
                "wall_s": 0.0,
            }
            for t in tasks
        ],
    }


def save_repeat(session: dict, root: Path = LABELS_DIR) -> Path:
    errs = [
        f"{t['clip_id']} frame {t['frame']}: {e}"
        for t in session["tasks"]
        for e in validate_points(t.get("points", {}), t["image_size"])
    ]
    if errs:
        raise ValueError("; ".join(errs))
    for t in session["tasks"]:
        t["points"] = _round_points(t.get("points", {}))
    p = repeat_dir(root) / f"{session['session_id']}.json"
    _write_json(p, session)
    return p
