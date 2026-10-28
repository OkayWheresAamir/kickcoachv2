import importlib.util
import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from kickcoach.labels import qa, schema, timing
from kickcoach.pose import cache, skeleton

W, H = 480, 360


def pt(x, y, v=2):
    return {"x": x, "y": y, "v": v}


def standing(dx=0.0, dy=0.0):
    """A plausible side-on figure: thigh 50 px, shank 50 px, both legs, ball at the feet."""
    pts = {}
    for side, off in (("left", -4.0), ("right", 4.0)):
        x = 200 + off + dx
        pts[f"{side}_shoulder"] = pt(x, 100 + dy)
        pts[f"{side}_hip"] = pt(x, 180 + dy)
        pts[f"{side}_knee"] = pt(x, 230 + dy)
        pts[f"{side}_ankle"] = pt(x, 280 + dy)
        pts[f"{side}_heel"] = pt(x - 6, 286 + dy)
        pts[f"{side}_big_toe"] = pt(x + 14, 288 + dy)
    pts["ball"] = pt(240 + dx, 280 + dy)
    return pts


def prefill_of(points):
    return {n: (None if p["v"] == 0 else [p["x"], p["y"], 0.9]) for n, p in points.items()}


def frame_entry(points, prefill=None, confirmed=True, when="2026-10-20T10:00:00+00:00"):
    return {
        "points": points,
        "prefill": prefill if prefill is not None else prefill_of(points),
        "confirmed": confirmed,
        "confirmed_at": when if confirmed else None,
        "active_s": 30.0,
        "wall_s": 40.0,
        "edits": 1,
    }


def full_label(cid="c01", purpose="eval"):
    lab = schema.new_label(cid, 100, W, H, purpose)
    lab["clip"].update(
        kicking_leg="right",
        kick_direction="right",
        view_ok_human="y",
        contact_frame=50,
        max_backswing_frame=40,
    )
    lab["clip_timing"] = {"active_s": 60.0, "wall_s": 90.0}
    for f in (40, 49, 50):
        lab["frames"][str(f)] = frame_entry(standing())
    return lab


def codes(flags):
    return [f["code"] for f in flags]


# --- schema ---------------------------------------------------------------------------


def test_frame_roles_are_backswing_contact_minus_one_and_contact():
    clip = {"max_backswing_frame": 40, "contact_frame": 50}
    assert schema.frame_roles(clip, 100) == {
        40: ["max_backswing"],
        49: ["pre_contact"],
        50: ["contact"],
    }


def test_frame_roles_skip_frames_outside_the_clip():
    assert schema.frame_roles({"max_backswing_frame": None, "contact_frame": 0}, 10) == {
        0: ["contact"]
    }


def test_prefill_points_maps_missing_model_points_to_none():
    row = skeleton.empty(1)[0]
    row[skeleton.INDEX["left_knee"]] = (10.0, 20.0, 0.5)
    pre = schema.prefill_points(row)
    assert pre["left_knee"] == [10.0, 20.0, 0.5]
    assert pre["right_knee"] is None and pre["ball"] is None
    assert set(pre) == set(schema.ALL_POINTS)


def test_points_array_marks_unknown_points_nan_with_v0():
    arr = schema.points_array({"left_hip": pt(1.0, 2.0, 1), "ball": pt(None, None, 0)})
    assert arr.shape == (len(schema.ALL_POINTS), 3)
    i = schema.ALL_POINTS.index("left_hip")
    assert tuple(arr[i]) == (1.0, 2.0, 1.0)
    b = arr[schema.ALL_POINTS.index("ball")]
    assert np.isnan(b[0]) and b[2] == 0


def test_valid_label_has_no_errors():
    assert schema.validate(full_label()) == []


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda lab: lab["clip"].update(kicking_leg="both"), "kicking_leg"),
        (lambda lab: lab["clip"].update(contact_frame=100), "contact_frame"),
        (lambda lab: lab["frames"].update({"100": frame_entry(standing())}), "frames key"),
        (lambda lab: lab["frames"]["50"]["points"].update(ball=pt(1.0, 1.0, 0)), "no coordinates"),
        (lambda lab: lab["frames"]["50"]["points"].update(ball=pt(None, None, 2)), "numeric"),
        (lambda lab: lab["frames"]["50"]["points"].update(ball=pt(W + 1.0, 5.0)), "outside"),
        (lambda lab: lab["frames"]["50"]["points"].update(nose=pt(5.0, 5.0)), "unknown point"),
        (lambda lab: lab["frames"]["50"]["points"].update(ball=pt(5.0, 5.0, 3)), "v must be"),
    ],
)
def test_validate_rejects_bad_values(mutate, fragment):
    lab = full_label()
    mutate(lab)
    errs = schema.validate(lab)
    assert any(fragment in e for e in errs), errs


def test_save_refuses_invalid_and_round_trips_valid(tmp_path):
    lab = full_label()
    lab["frames"]["50"]["points"]["ball"] = pt(240.123456, 280.987654)
    p = schema.save(lab, tmp_path)
    back = schema.load("c01", tmp_path)
    assert back["frames"]["50"]["points"]["ball"] == pt(240.12, 280.99)
    assert not list(tmp_path.glob("*.tmp"))
    lab["clip"]["view_ok_human"] = "maybe"
    with pytest.raises(ValueError, match="view_ok_human"):
        schema.save(lab, tmp_path)
    assert schema.load("c01", tmp_path)["clip"]["view_ok_human"] == "y"  # file untouched
    assert p.name == "c01.json"


def test_status_eval_and_gate_only():
    lab = full_label()
    assert schema.status(lab)["complete"]
    del lab["frames"]["49"]  # contact-1 is optional (DEC-014)
    assert schema.status(lab)["complete"]
    lab["frames"]["50"]["confirmed"] = False
    st = schema.status(lab)
    assert not st["complete"] and st["frames_confirmed"] == 1 and st["frames_needed"] == 2
    gate = schema.new_label("c26", 50, W, H, "gate_only")
    assert not schema.status(gate)["complete"]
    gate["clip"]["view_ok_human"] = "n"
    assert schema.status(gate)["complete"]


# --- QA -------------------------------------------------------------------------------


def test_complete_consistent_label_has_no_flags():
    assert qa.flags(full_label()) == []


def test_missing_clip_fields_and_unconfirmed_frames_are_todo():
    lab = full_label()
    lab["clip"]["kick_direction"] = None
    lab["frames"]["49"]["confirmed"] = False  # optional frame, but started: still flagged
    del lab["frames"]["50"]  # required frame not started
    fl = qa.flags(lab)
    assert [(f["code"], f["frame"]) for f in fl] == [
        ("clip_incomplete", None),
        ("frame_unconfirmed", 49),
        ("frame_unconfirmed", 50),
    ]
    assert all(f["severity"] == "todo" for f in fl)
    del lab["frames"]["49"]  # optional frame never opened: no flag
    assert ("frame_unconfirmed", 49) not in [(f["code"], f["frame"]) for f in qa.flags(lab)]


def test_backswing_after_contact_is_flagged():
    lab = full_label()
    lab["clip"]["max_backswing_frame"] = 55
    assert "event_order" in codes(qa.clip_flags(lab))


def test_missing_required_point_and_ball_on_contact():
    pts = standing()
    pts["right_knee"] = pt(None, None, 0)  # kicking leg
    pts["left_knee"] = pt(None, None, 0)  # plant-leg knee is not required
    pts["ball"] = pt(None, None, 0)
    fl = qa.frame_flags(pts, prefill_of(standing()), "right", 50, ["contact"])
    missing = {f["point"] for f in fl if f["code"] == "missing_required"}
    assert missing == {"right_knee", "ball"}


def test_far_from_model_uses_thigh_lengths():
    pts = standing()
    model = prefill_of(standing())
    pts["right_heel"] = pt(pts["right_heel"]["x"] + 30, pts["right_heel"]["y"])  # 0.6 thigh
    pts["left_heel"] = pt(pts["left_heel"]["x"] + 20, pts["left_heel"]["y"])  # 0.4 thigh
    fl = qa.frame_flags(pts, model, "right", 50, [])
    assert [(f["code"], f["point"]) for f in fl] == [("far_from_model", "right_heel")]


def test_left_right_swap_relative_to_model_is_named_as_a_swap():
    model = standing()
    pts = standing()
    # Model put the knees 40 px apart; the labeller has them the other way round.
    model["left_knee"], model["right_knee"] = pt(180.0, 230.0), pt(220.0, 230.0)
    pts["left_knee"], pts["right_knee"] = pt(220.0, 230.0), pt(180.0, 230.0)
    fl = qa.frame_flags(pts, prefill_of(model), "right", 50, [])
    assert codes(fl) == ["lr_swap_vs_model"]


def test_segment_length_change_beyond_tolerance_is_flagged():
    lab = full_label()
    knee = lab["frames"]["50"]["points"]["right_knee"]
    knee["y"] = 245.0  # right thigh 65 px here vs 50 px elsewhere: +30%
    fl = [f for f in qa.flags(lab) if f["code"] == "segment_length"]
    assert {f["message"].split(" is ")[0] for f in fl} == {"right thigh", "right shank"}
    knee["y"] = 234.0  # +8%, inside the tolerance
    assert not [f for f in qa.flags(lab) if f["code"] == "segment_length"]


def test_gate_only_clip_needs_only_view_ok():
    gate = schema.new_label("c26", 50, W, H, "gate_only")
    assert codes(qa.flags(gate)) == ["clip_incomplete"]
    gate["clip"]["view_ok_human"] = "n"
    assert qa.flags(gate) == []


# --- repeatability sampling and timing ------------------------------------------------


def test_repeat_sampling_spreads_over_clips_and_skips_fresh_and_gate_frames():
    now = datetime.fromisoformat("2026-10-30T12:00:00+00:00")
    old = (now - timedelta(hours=72)).isoformat()
    fresh = (now - timedelta(hours=1)).isoformat()
    labels = {}
    for cid in ("c01", "c02", "c03"):
        lab = full_label(cid)
        for e in lab["frames"].values():
            e["confirmed_at"] = old
        labels[cid] = lab
    labels["c03"]["frames"]["50"]["confirmed_at"] = fresh
    gate = schema.new_label("c26", 50, W, H, "gate_only")
    gate["frames"]["10"] = frame_entry(standing(), when=old)
    labels["c26"] = gate

    tasks = schema.sample_repeat_tasks(labels, 6, seed=1, min_age_hours=48, now=now)
    assert len(tasks) == 6
    assert {t["clip_id"] for t in tasks} == {"c01", "c02", "c03"}
    assert {"clip_id": "c03", "frame": 50} not in tasks
    assert all(sum(t["clip_id"] == c for t in tasks) == 2 for c in ("c01", "c02", "c03"))
    again = schema.sample_repeat_tasks(labels, 6, seed=1, min_age_hours=48, now=now)
    assert tasks == again
    assert len(schema.sample_repeat_tasks(labels, 50, 1, 48, now)) == 8  # all eligible


def test_save_repeat_validates_points(tmp_path):
    s = schema.new_repeat_session("r1", 1, [{"clip_id": "c01", "frame": 5}])
    s["tasks"][0]["image_size"] = [W, H]
    s["tasks"][0]["points"] = {"ball": pt(W + 50.0, 3.0)}
    with pytest.raises(ValueError, match="outside"):
        schema.save_repeat(s, tmp_path)
    s["tasks"][0]["points"] = {"ball": pt(10.0, 3.0)}
    assert schema.save_repeat(s, tmp_path).exists()


def test_timing_summary_and_projection():
    labels = {"c01": full_label("c01"), "c02": full_label("c02")}
    labels["c02"]["clip_timing"]["active_s"] = 100.0
    s = timing.summary(labels)
    assert s["n_frames"] == 6 and s["frame_s_median"] == 30.0
    assert s["n_clips"] == 2 and s["events_s_median"] == 80.0
    assert timing.projection_minutes(80.0, 30.0, 23, 15, 2) == pytest.approx(
        (23 * 80 + 15 * 2 * 30) / 60
    )
    assert timing.summary({}) is None


# --- the HTTP tool, end to end on a tmp copy of the data layout -------------------------


def load_tool():
    p = Path(__file__).resolve().parents[1] / "tools" / "label_tool.py"
    spec = importlib.util.spec_from_file_location("label_tool", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def server(tmp_path):
    tool = load_tool()
    frames, pose, labels = tmp_path / "frames", tmp_path / "pose", tmp_path / "labels"
    (frames / "c01").mkdir(parents=True)
    meta = {"n_frames": 3, "stream": {"width": W, "height": H}}
    (frames / "c01" / "meta.json").write_text(json.dumps(meta))
    (frames / "c01" / "000002.jpg").write_bytes(b"\xff\xd8fakejpeg")
    poses = skeleton.empty(3)
    poses[:, skeleton.INDEX["left_knee"]] = (100.0, 200.0, 0.8)
    cache.save(schema.PREFILL_MODEL, "c01", poses, {}, root=pose)
    manifest = {"c01": {"clip_id": "c01", "purpose": "eval", "split": "tune"}}

    def start(repeat=None):
        app = tool.App(["c01"], manifest, frames, pose, labels, repeat)
        srv = tool.ThreadingHTTPServer(("127.0.0.1", 0), tool.make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    servers = []
    yield start, labels
    for s in servers:
        s.shutdown()


def call(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_tool_serves_clip_and_saves_labels(server):
    start, labels_dir = server
    base = start()
    code, body = call(f"{base}/api/clip/c01")
    assert code == 200
    clip = json.loads(body)
    i = schema.ALL_POINTS.index("left_knee")
    assert clip["prefill"][0][i] == [100.0, 200.0, 0.800000011920929]
    assert clip["label"]["clip"]["contact_frame"] is None

    lab = clip["label"]
    lab["clip"]["contact_frame"] = 2
    lab["n_frames"] = 999  # the page can't change what the tool owns
    code, body = call(f"{base}/api/clip/c01", "PUT", lab)
    assert code == 200 and json.loads(body)["ok"]
    saved = schema.load("c01", labels_dir)
    assert saved["clip"]["contact_frame"] == 2 and saved["n_frames"] == 3

    lab["clip"]["contact_frame"] = 7  # out of range: refused, file unchanged
    code, body = call(f"{base}/api/clip/c01", "PUT", lab)
    assert code == 400 and "contact_frame" in json.loads(body)["error"]
    assert schema.load("c01", labels_dir)["clip"]["contact_frame"] == 2

    assert call(f"{base}/frames/c01/2.jpg")[0] == 200
    assert call(f"{base}/frames/c01/5.jpg")[0] == 404
    assert call(f"{base}/api/clip/c99")[0] == 404
    assert call(f"{base}/static/../label_tool.py")[0] == 404


def test_repeat_mode_never_sends_first_labels(server):
    start, _ = server
    session = schema.new_repeat_session("r1", 1, [{"clip_id": "c01", "frame": 2}])
    session["tasks"][0]["image_size"] = [W, H]
    base = start(repeat=session)
    assert call(f"{base}/api/clip/c01")[0] == 403
    code, body = call(f"{base}/api/repeat")
    task = json.loads(body)["tasks"][0]
    assert code == 200 and task["frame"] == 2 and task["n_frames"] == 3
    assert "label" not in task and task["points"] == {}
