"""SQLite sessions and self-history deltas."""

from __future__ import annotations

from kickcoach.storage import db


def result(knee: float | None, accepted: bool = True) -> dict:
    feats = [
        {"id": "knee_angle_contact", "value": knee, "unit": "deg",
         "status": "measured" if knee is not None else "not_measured"},
        {"id": "segment_sequence_order", "value": "proximal_to_distal", "unit": "order",
         "status": "measured"},
    ]  # fmt: skip
    return {
        "quality": {"accepted": accepted, "reasons": []},
        "events": {"contact_frame": 10},
        "features": feats if accepted else [],
        "faults": [{"fault_id": "F-LEAN-BACK", "evidence": {"trunk_lean_contact": -12.0},
                    "threshold_source": "heuristic: test"}],
        "feedback": {"items": [], "generated_by": "template", "validator_passed": None,
                     "attempts": [], "fallback_reason": "x"},
    }  # fmt: skip


def save(con, player, knee, **kw):
    return db.save_session(con, player=player, level=None, clip_name="k.mp4",
                           result=result(knee, **kw), inputs={})  # fmt: skip


def test_history_delta_is_against_the_last_three_measured_sessions(tmp_path):
    con = db.connect(tmp_path / "t.db")
    for knee in (100.0, 120.0, 130.0, 140.0):
        save(con, "ana", knee)
    save(con, "ana", None)  # not measured: skipped
    save(con, "bo", 10.0)  # another player: ignored
    sid = save(con, "ana", 150.0)
    d = db.history_delta(con, "ana", sid)
    assert d["knee_angle_contact"]["delta"] == 20.0  # 150 - mean(120, 130, 140)
    assert d["knee_angle_contact"]["text"] == "+20.0 vs last 3 sessions"
    assert "segment_sequence_order" not in d  # not numeric


def test_first_session_has_no_delta_and_listing(tmp_path):
    con = db.connect(tmp_path / "t.db")
    sid = save(con, "ana", 140.0)
    assert db.history_delta(con, "ana", sid) == {}
    save(con, "ana", None, accepted=False)
    rows = db.list_sessions(con, "ana")
    assert [r["accepted"] for r in rows] == [False, True]
    assert rows[1]["features"]["knee_angle_contact"] == 140.0
    assert rows[1]["faults"] == ["F-LEAN-BACK"]
    assert con.execute("SELECT count(*) FROM feedback").fetchone()[0] == 2
