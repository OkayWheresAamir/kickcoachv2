"""QA flags on human labels, shown in the labelling tool. A flag asks Aamir to look again;
nothing here changes a label (DEC-007).

Severity "todo" = something still to do; "check" = a value worth a second look, which may
well be right (e.g. a big correction of a model mistake).
"""

from __future__ import annotations

import numpy as np

from kickcoach.labels.schema import (
    ALL_POINTS,
    BALL,
    SIDES,
    UNKNOWN,
    frame_roles,
    points_array,
    required_frames,
)

# heuristic: in 2D, a thigh or shank looks shorter when it turns out of the image plane, so
# some change across frames is real. Only a change bigger than this is flagged.
SEGMENT_RATIO_TOL = 0.20
# heuristic: a label this far from the model pre-fill, in thigh lengths, gets a second look.
# For scale, E1's PCK threshold is 0.1 thigh lengths.
FAR_FROM_MODEL_THIGHS = 0.5
# heuristic: a left/right pair counts as swapped relative to the model when the crossed
# match is under this fraction of the straight match.
SWAP_RATIO = 0.5

_IDX = {name: i for i, name in enumerate(ALL_POINTS)}
_JOINTS = ("shoulder", "hip", "knee", "ankle", "heel", "big_toe")


def flag(
    code: str, severity: str, message: str, frame: int | None = None, point: str | None = None
) -> dict:
    return {"code": code, "severity": severity, "message": message, "frame": frame, "point": point}


def required_points(kicking_leg: str | None) -> tuple[str, ...]:
    """Points a labelled frame needs for the core features: trunk lean (both shoulders and
    hips), the kicking-leg knee angle, and the plant-foot ankle."""
    if kicking_leg not in SIDES:
        return tuple(f"{s}_{j}" for j in ("shoulder", "hip", "knee", "ankle") for s in SIDES)
    plant = "left" if kicking_leg == "right" else "right"
    return (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        f"{kicking_leg}_knee",
        f"{kicking_leg}_ankle",
        f"{plant}_ankle",
    )


def _xy(arr: np.ndarray, name: str) -> np.ndarray | None:
    p = arr[_IDX[name]]
    return None if p[2] == UNKNOWN else p[:2]


def _prefill_array(prefill: dict) -> np.ndarray:
    return points_array({k: {"x": v[0], "y": v[1], "v": 2} for k, v in prefill.items() if v})


def thigh_length(arr: np.ndarray) -> float | None:
    lengths = []
    for s in SIDES:
        hip, knee = _xy(arr, f"{s}_hip"), _xy(arr, f"{s}_knee")
        if hip is not None and knee is not None:
            lengths.append(float(np.linalg.norm(hip - knee)))
    return float(np.mean(lengths)) if lengths and np.mean(lengths) > 0 else None


def clip_flags(label: dict) -> list[dict]:
    clip = label["clip"]
    if label.get("purpose") == "gate_only":
        if clip["view_ok_human"] is None:
            return [flag("clip_incomplete", "todo", "view OK (y/n) is not set")]
        return []
    out = [
        flag("clip_incomplete", "todo", f"{k.replace('_', ' ')} is not set")
        for k in (
            "kicking_leg",
            "kick_direction",
            "view_ok_human",
            "contact_frame",
            "max_backswing_frame",
        )
        if clip[k] is None
    ]
    b, c = clip["max_backswing_frame"], clip["contact_frame"]
    if b is not None and c is not None and b >= c:
        out.append(
            flag(
                "event_order",
                "check",
                f"max backswing (frame {b}) is not before contact (frame {c})",
            )
        )
    return out


def frame_flags(
    points: dict, prefill: dict, kicking_leg: str | None, frame: int, roles: list[str]
) -> list[dict]:
    out = []
    arr = points_array(points)
    for name in required_points(kicking_leg):
        if _xy(arr, name) is None:
            out.append(
                flag(
                    "missing_required",
                    "check",
                    f"{name} is not placed; it's needed for the angles. Mark it "
                    "occluded with your best estimate if you can judge it",
                    frame,
                    name,
                )
            )
    if "contact" in roles and _xy(arr, BALL) is None:
        out.append(
            flag(
                "missing_required",
                "check",
                "ball is not placed on the contact frame (ignore if it can't be seen)",
                frame,
                BALL,
            )
        )
    model = _prefill_array(prefill)
    thigh = thigh_length(arr) or thigh_length(model)
    if thigh is None:
        return out
    far = FAR_FROM_MODEL_THIGHS * thigh
    swapped: set[str] = set()
    for j in _JOINTS:
        lname, rname = f"left_{j}", f"right_{j}"
        L, R, mL, mR = _xy(arr, lname), _xy(arr, rname), _xy(model, lname), _xy(model, rname)
        if any(p is None for p in (L, R, mL, mR)):
            continue
        straight = np.linalg.norm(L - mL) + np.linalg.norm(R - mR)
        crossed = np.linalg.norm(L - mR) + np.linalg.norm(R - mL)
        if straight > far and crossed < SWAP_RATIO * straight:
            swapped |= {lname, rname}
            out.append(
                flag(
                    "lr_swap_vs_model",
                    "check",
                    f"left/right {j.replace('_', ' ')} are swapped relative to the model. "
                    "Usually the model is wrong; check which leg is which",
                    frame,
                    lname,
                )
            )
    for name in ALL_POINTS:
        p, m = _xy(arr, name), _xy(model, name)
        if name in swapped or p is None or m is None:
            continue
        d = float(np.linalg.norm(p - m))
        if d > far:
            out.append(
                flag(
                    "far_from_model",
                    "check",
                    f"{name} is {d / thigh:.1f} thigh lengths from the model's guess",
                    frame,
                    name,
                )
            )
    return out


def segment_length_flags(label: dict) -> list[dict]:
    out = []
    frames = sorted((int(k), points_array(e["points"])) for k, e in label["frames"].items())
    for s in SIDES:
        for seg, (a, b) in (("thigh", ("hip", "knee")), ("shank", ("knee", "ankle"))):
            lengths = []
            for idx, arr in frames:
                pa, pb = _xy(arr, f"{s}_{a}"), _xy(arr, f"{s}_{b}")
                if pa is not None and pb is not None:
                    lengths.append((float(np.linalg.norm(pa - pb)), idx))
            if len(lengths) < 2 or min(lengths)[0] <= 0:
                continue
            (lo, f_lo), (hi, f_hi) = min(lengths), max(lengths)
            if hi / lo > 1 + SEGMENT_RATIO_TOL:
                out.append(
                    flag(
                        "segment_length",
                        "check",
                        f"{s} {seg} is {lo:.0f} px in frame {f_lo} but {hi:.0f} px in "
                        f"frame {f_hi} ({hi / lo - 1:.0%} longer); check the {s} "
                        f"{a}/{b} points",
                        f_hi,
                        f"{s}_{b}",
                    )
                )
    return out


def flags(label: dict) -> list[dict]:
    """Every QA flag for one clip label, clip-level first, then by frame."""
    out = clip_flags(label)
    if label.get("purpose") == "gate_only":
        return out
    roles = frame_roles(label["clip"], label["n_frames"])
    required = set(required_frames(label["clip"], label["n_frames"]))
    for idx in sorted(roles):
        started = str(idx) in label["frames"]
        if (idx in required or started) and not label["frames"].get(str(idx), {}).get("confirmed"):
            out.append(
                flag(
                    "frame_unconfirmed",
                    "todo",
                    f"{' + '.join(roles[idx])} frame {idx} is not confirmed yet",
                    idx,
                )
            )
    for key in sorted(label["frames"], key=int):
        e = label["frames"][key]
        out += frame_flags(
            e["points"],
            e.get("prefill", {}),
            label["clip"]["kicking_leg"],
            int(key),
            roles.get(int(key), []),
        )
    return out + segment_length_flags(label)
