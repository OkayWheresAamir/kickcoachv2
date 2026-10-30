import numpy as np
import pytest

from eval import e1_pose
from kickcoach.labels import schema
from kickcoach.pose import skeleton

# A right-footed kicker facing image right; thigh = 100 px on both sides.
BASE = {
    "left_shoulder": (200, 100), "right_shoulder": (205, 100),
    "left_hip": (200, 300), "right_hip": (205, 300),
    "left_knee": (200, 400), "right_knee": (205, 400),
    "left_ankle": (200, 500), "right_ankle": (150, 480),
    "left_heel": (190, 510), "right_heel": (140, 470),
    "left_big_toe": (230, 510), "right_big_toe": (130, 510),
}  # fmt: skip


def _label(points=BASE, prefill=None, v=schema.VISIBLE):
    pts = {n: {"x": float(x), "y": float(y), "v": v} for n, (x, y) in points.items()}
    pre = prefill if prefill is not None else {n: [x, y, 0.9] for n, (x, y) in points.items()}
    return {
        "purpose": "eval",
        "n_frames": 10,
        "clip": {
            "kicking_leg": "right",
            "kick_direction": "right",
            "contact_frame": 5,
            "max_backswing_frame": 2,
        },
        "frames": {"5": {"confirmed": True, "points": pts, "prefill": pre}},
    }


def _pose(points, shift=(0.0, 0.0), drop=()):
    arr = skeleton.empty(10)
    for n, (x, y) in points.items():
        if n not in drop:
            arr[5, skeleton.INDEX[n]] = (x + shift[0], y + shift[1], 0.9)
    return arr


def _run(poses, label=None):
    labels = {"c01": label or _label()}
    manifest = {"c01": {"split": "tune"}}
    return e1_pose.point_rows(labels, manifest, {m: {"c01": p} for m, p in poses.items()})


def test_pck_counts_within_threshold_and_missing_as_miss():
    rows = _run(
        {
            "movenet": _pose(BASE, shift=(15, 0)),  # 0.15 thigh: outside 0.1
            "rtmpose": _pose(BASE),  # exact
            "mediapipe": _pose(BASE, shift=(5, 0), drop=("right_big_toe",)),  # 0.05 thigh
        }
    )
    s = {m: e1_pose.summarise(e1_pose.select(rows, model=m)) for m in e1_pose.MODELS}
    assert s["rtmpose"]["pck"] == 1.0 and s["rtmpose"]["missing_rate"] == 0.0
    assert s["movenet"]["pck"] == 0.0
    assert s["movenet"]["median_err"] == pytest.approx(0.15)
    assert s["mediapipe"]["missing_rate"] == pytest.approx(1 / 12)
    assert s["mediapipe"]["pck"] == pytest.approx(11 / 12)
    assert s["mediapipe"]["median_err"] == pytest.approx(0.05)  # over predicted points only


def test_groups_follow_the_kicking_leg():
    assert e1_pose.group_of("right_knee", "right") == "kicking_leg"
    assert e1_pose.group_of("left_heel", "right") == "plant_foot"
    assert e1_pose.group_of("right_big_toe", "right") == "kicking_foot"
    assert e1_pose.group_of("left_shoulder", "left") == "shoulders"


def test_unmoved_points_are_detected_from_the_prefill():
    pre = {n: [x, y, 0.9] for n, (x, y) in BASE.items()}
    pre["right_knee"] = [215.0, 400.0, 0.9]  # the labeller moved this one by 10 px
    pre["right_ankle"] = [150.004, 480.0, 0.9]  # only label rounding: untouched
    rows = _run({m: _pose(BASE) for m in e1_pose.MODELS}, _label(prefill=pre))
    rt = {r["point"]: r for r in e1_pose.select(rows, model="rtmpose")}
    assert not rt["right_knee"]["unmoved"]
    assert rt["right_knee"]["moved_px"] == pytest.approx(10.0)
    assert rt["right_ankle"]["unmoved"]


def test_unknown_points_are_not_scored():
    label = _label()
    label["frames"]["5"]["points"]["left_heel"] = {"v": schema.UNKNOWN}
    rows = _run({m: _pose(BASE) for m in e1_pose.MODELS}, label)
    assert "left_heel" not in {r["point"] for r in rows}


def test_leg_swap_is_detected():
    # Legs far apart, so the swap is unambiguous.
    far = {**BASE, "right_knee": (300, 380), "right_ankle": (380, 420)}
    far_swapped = {**far}
    for j in ("knee", "ankle"):
        far_swapped[f"left_{j}"], far_swapped[f"right_{j}"] = far[f"right_{j}"], far[f"left_{j}"]
    labels = {"c01": _label(points=far)}
    poses = {
        "movenet": {"c01": _pose(far_swapped)},
        "rtmpose": {"c01": _pose(far)},
        "mediapipe": {"c01": _pose(far)},
    }
    swaps = e1_pose.leg_swaps(labels, poses)
    assert swaps["movenet"] == [("c01", 5)]
    assert not swaps.get("rtmpose")


def test_frame_angles_match_geometry():
    pts = {n: np.array(p, dtype=float) for n, p in BASE.items()}
    a = e1_pose.frame_angles(pts, "left", facing=1)
    assert a["knee_angle"] == pytest.approx(180.0)  # left hip, knee, ankle are collinear
    assert a["trunk_lean"] == pytest.approx(0.0)  # shoulders straight above the hips


def test_bootstrap_lead_of_a_perfect_model():
    rows = _run({"movenet": _pose(BASE, shift=(15, 0)), "rtmpose": _pose(BASE),
                 "mediapipe": _pose(BASE, shift=(15, 0))})  # fmt: skip
    boot = e1_pose.clip_bootstrap_pck(rows, ("rtmpose", "mediapipe"), n_boot=50)
    assert np.all(boot["rtmpose"] - boot["mediapipe"] == 1.0)
