"""E2, E3, E4 and E8 scoring logic, on hand-made inputs (no clips needed)."""

import numpy as np
import pytest

from eval import e2_angles, e3_contact, e4_gemini, e8_reliability
from eval.e4_prompt import SCHEMA, SCHEMA_V1, build_prompt
from kickcoach.labels import schema

# A right-footed kicker facing image right: knee 90 deg at contact (frame 5), straight leg
# at max backswing (frame 2), trunk vertical.
PTS = {
    "left_shoulder": (200, 100),
    "right_shoulder": (205, 100),
    "left_hip": (200, 300),
    "right_hip": (205, 300),
    "left_knee": (200, 400),
    "right_knee": (205, 400),
    "left_ankle": (200, 500),
    "right_ankle": (305, 400),
}


def _label(prefill_shift: dict | None = None) -> dict:
    pts = {n: {"x": float(x), "y": float(y), "v": schema.VISIBLE} for n, (x, y) in PTS.items()}
    shift = prefill_shift or {}
    pre = {n: [x + shift.get(n, 0.0), y, 0.9] for n, (x, y) in PTS.items()}
    frame = {"confirmed": True, "points": pts, "prefill": pre}
    return {
        "purpose": "eval",
        "n_frames": 10,
        "clip": {
            "kicking_leg": "right",
            "kick_direction": "right",
            "contact_frame": 5,
            "max_backswing_frame": 2,
        },
        "frames": {"5": frame, "2": frame},
    }


def test_truth_uses_the_labelled_frame_and_counts_moved_points():
    tr = e2_angles.truth(_label(prefill_shift={"right_knee": 3.0}))
    assert tr["knee_angle_contact"]["truth"] == pytest.approx(90.0)
    assert tr["trunk_lean_contact"]["truth"] == pytest.approx(0.0)
    assert tr["knee_angle_contact"]["points_moved"] == 1  # the knee left its pre-fill
    assert tr["trunk_lean_contact"]["points_moved"] == 0  # anchored: raw RTMPose = truth


def test_error_rows_keep_unanswered_features_with_their_reason():
    tr = e2_angles.truth(_label())
    vals = {
        "knee_angle_contact": {"value": 95.0, "frame": 6},
        "max_knee_flexion_backswing": {"value": None, "reason": "no backswing"},
        "trunk_lean_contact": {"value": -2.0, "frame": 5},
    }
    rows = {r["feature"]: r for r in e2_angles.error_rows("s", vals, "c01", "tune", tr, "v")}
    assert rows["knee_angle_contact"]["err"] == pytest.approx(5.0)
    assert rows["max_knee_flexion_backswing"]["err"] is None
    assert rows["max_knee_flexion_backswing"]["reason"] == "no backswing"
    s = e2_angles.summary(list(rows.values()))
    assert (s["n"], s["answered"]) == (3, 2)
    assert s["mae"] == pytest.approx(3.5) and s["bias"] == pytest.approx(1.5)


def test_bootstrap_compares_systems_on_the_same_clips():
    rows = [{"system": "a", "clip_id": c, "err": e} for c, e in (("c1", 10.0), ("c2", -10.0))] + [
        {"system": "b", "clip_id": c, "err": e} for c, e in (("c1", 1.0), ("c2", None), ("c3", 5.0))
    ]
    b = e2_angles.bootstrap_mae(rows, ("a", "b"))
    assert b["n_clips"] == 1  # only c1 has both
    assert b["diff"] == pytest.approx(9.0)


def _e3_row(cid, split, frame, label, bracket=None, method="ball_uncorrected"):
    return {
        "clip_id": cid,
        "split": split,
        "method": method,
        "frame": frame,
        "label_frame": label,
        "err": None if frame is None else frame - label,
        "bracket": bracket,
    }


def test_contact_metrics_count_failures_as_misses():
    rows = [
        _e3_row("a", "tune", 10, 10),
        _e3_row("b", "tune", 12, 10),
        _e3_row("c", "tune", None, 10),
        _e3_row("d", "tune", 30, 10),
    ]
    m = e3_contact.metrics(rows)
    assert m["failed"] == 1 and m["gross"] == 1
    assert m["exact"] == pytest.approx(0.25) and m["within2"] == pytest.approx(0.5)
    assert m["mae"] == pytest.approx((0 + 2 + 20) / 3)


def test_bracket_shift_is_estimated_without_the_clip_and_kept_inside_the_bracket():
    rows = [
        _e3_row("t1", "tune", 8, 10, (5, 20)),
        _e3_row("t2", "tune", 8, 10, (5, 20)),
        _e3_row("t3", "tune", 18, 20, (15, 19)),
        _e3_row("h1", "heldout", 10, 10, (9, 10)),  # clamped: can't move
    ]
    b = e3_contact.bracket_analysis(rows)
    assert b["k"] == 2
    assert [r["after"] for r in b["loo"]] == [0, 0, -1]  # t3 is clamped at 19
    assert b["heldout"][0]["after"] == 0
    assert b["adopt"]


def test_bracket_shift_is_rejected_when_it_hurts_the_held_out_clips():
    rows = [
        _e3_row("t1", "tune", 8, 10, (5, 20)),
        _e3_row("t2", "tune", 8, 10, (5, 20)),
        _e3_row("h1", "heldout", 10, 10, (9, 20)),
    ]
    assert not e3_contact.bracket_analysis(rows)["adopt"]


def test_gate_confusion_treats_refusal_as_positive_and_skips_unlabelled():
    rows = [
        {"view_ok_human": "n", "r": True},
        {"view_ok_human": "n", "r": False},
        {"view_ok_human": "y", "r": True},
        {"view_ok_human": "y", "r": False},
        {"view_ok_human": None, "r": True},
    ]
    c = e8_reliability.confusion(rows, "r")
    assert (c["n"], c["tp"], c["fp"], c["fn"], c["tn"]) == (4, 1, 1, 1, 1)
    assert c["precision"] == pytest.approx(0.5) and c["recall"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("meta", "allowed"),
    [
        ({"fps_trusted": True, "seconds_per_frame": 1 / 300}, True),
        ({"fps_trusted": True, "seconds_per_frame": 1 / 30}, False),  # real time, too slow
        ({"fps_trusted": False, "seconds_per_frame": 1 / 300}, False),
    ],
)
def test_velocity_needs_a_trusted_rate_of_120_fps(meta, allowed):
    assert e8_reliability.velocity_allowed(meta) is allowed


def test_gemini_answer_is_converted_to_frames_and_flags():
    rec = {
        "key": "k",
        "parsed": {
            "kicking_leg": "right",
            "contact_time_s": 2.5,
            "knee_angle_contact_deg": 150,
            "trunk_lean_contact_deg": True,
            "max_backswing_time_s": None,
            "max_knee_flexion_deg": 80.0,
            "cannot_determine_reason": None,
            "notes": "Contact at approximately 2.5 s.",
        },
        "response": {"usage_metadata": {"prompt_token_count": 10}},
    }
    a = e4_gemini.parse_answer(rec, fps=30.0)
    assert a["contact_frame"] == 75 and a["backswing_frame"] is None
    assert a["trunk_lean_contact_deg"] is None  # a bool is not a number
    assert a["n_numbers"] == 3 and a["hedged"]


def test_unparseable_gemini_answer_has_no_numbers():
    a = e4_gemini.parse_answer({"key": "k", "parsed": None, "response": {}}, fps=30.0)
    assert not a["parsed_ok"] and a["n_numbers"] == 0 and not a["hedged"]


def test_e4_v2_asks_for_every_e2_angle_and_keeps_v1_fields():
    assert set(SCHEMA_V1["required"]) < set(SCHEMA["required"])
    assert {"max_knee_flexion_deg", "max_backswing_time_s"} <= set(SCHEMA["required"])
    p = build_prompt(5.0, "left")
    assert "left leg" in p and "Maximum backswing" in p and "cannot_determine_reason" in p
    assert np.isclose(5.0, float(p.split("(")[1].split(" s")[0]))
