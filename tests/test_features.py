"""Synthetic-kick tests for the measurement core: known angles, contact frame and velocity,
with noise, at several frame rates (eval plan, "Synthetic kinematics")."""

import numpy as np
import pytest
from synthetic_kick import ball_detections, make_kick

from kickcoach.features.bounds import BOUNDS, violation
from kickcoach.pipeline import analyse
from kickcoach.pose import skeleton


def _meta(fps, trusted):
    return {
        "fps_trusted": trusted,
        "seconds_per_frame": 1 / fps if trusted else None,
        "fps_reason": "test",
        "constant_frame_rate": True,
        "repeated_frame_ratio": 0.0,
    }


def _run(k, trusted=True, ball=True):
    res = analyse(k.poses, _meta(k.fps, trusted), ball_detections(k) if ball else None)
    return res, {f["id"]: f for f in res["features"]}


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
@pytest.mark.parametrize("facing", [1, -1])
def test_noise_free_kick_recovers_the_known_values(fps, facing):
    k = make_kick(fps, facing=facing)
    res, f = _run(k)
    assert res["quality"]["accepted"], res["quality"]["reasons"]
    assert res["events"]["contact_frame"] == k.contact_frame
    assert res["events"]["kick_direction"] == ("right" if facing == 1 else "left")
    assert f["knee_angle_contact"]["value"] == pytest.approx(k.knee_contact, abs=1.0)
    assert f["max_knee_flexion_backswing"]["value"] == pytest.approx(k.knee_min, abs=1.5)
    assert f["trunk_lean_contact"]["value"] == pytest.approx(k.trunk_contact, abs=0.5)
    assert f["plant_foot_offset"]["value"] == pytest.approx(k.plant_offset, abs=0.02)
    assert f["segment_sequence_order"]["value"] == "proximal_to_distal"


@pytest.mark.parametrize("fps", [60, 120, 240])
def test_noisy_kick_stays_within_tolerance(fps):
    errs = {"knee_angle_contact": [], "max_knee_flexion_backswing": [], "trunk_lean_contact": []}
    for seed in range(5):
        k = make_kick(fps, noise_px=1.5, seed=seed)
        _, f = _run(k)
        errs["knee_angle_contact"].append(f["knee_angle_contact"]["value"] - k.knee_contact)
        errs["max_knee_flexion_backswing"].append(
            f["max_knee_flexion_backswing"]["value"] - k.knee_min
        )
        errs["trunk_lean_contact"].append(f["trunk_lean_contact"]["value"] - k.trunk_contact)
    # 1.5 px noise on 100 px segments; tolerances are for the mean abs error over 5 seeds.
    assert np.mean(np.abs(errs["knee_angle_contact"])) < 2.5
    assert np.mean(np.abs(errs["max_knee_flexion_backswing"])) < 3.0
    assert np.mean(np.abs(errs["trunk_lean_contact"])) < 1.0


def test_velocity_needs_a_trusted_frame_rate_of_at_least_120():
    k = make_kick(240, noise_px=0.5)
    _, trusted = _run(k)
    v = trusted["peak_knee_ext_velocity"]
    assert v["status"] == "measured"
    assert v["value"] == pytest.approx(k.peak_knee_ext_vel, rel=0.15)
    _, untrusted = _run(k, trusted=False)
    assert untrusted["peak_knee_ext_velocity"]["status"] == "not_measured"
    assert "unknown" in untrusted["peak_knee_ext_velocity"]["reason"]
    _, slow = _run(make_kick(60))
    assert "below 120" in slow["peak_knee_ext_velocity"]["reason"]


def test_lag_in_ms_only_with_trusted_timing():
    k = make_kick(240)
    _, f = _run(k)
    assert f["segment_sequence_order"]["extra"]["lag_ms"] > 0
    _, g = _run(k, trusted=False)
    assert g["segment_sequence_order"]["value"] == "proximal_to_distal"
    assert g["segment_sequence_order"]["extra"]["lag_ms"] is None
    _, slow = _run(make_kick(30))  # trusted, but too slow to time a lag (handoff D6)
    assert slow["segment_sequence_order"]["extra"]["lag_ms"] is None


def test_the_contact_break_protects_the_peak_velocity():
    from kickcoach.events.kick import knee_angle
    from kickcoach.signal import clean, derivative

    k = make_kick(240)
    c, b = k.contact_frame, k.max_flex_frame
    peaks = {}
    for name, breaks in (("through", ()), ("break", (c,))):
        xy = clean(k.poses, seconds_per_frame=1 / 240, breaks=breaks).xy
        w = derivative(knee_angle(xy, "right"), dt=1 / 240, breaks=breaks)
        peaks[name] = np.nanmax(w[b : c + 1])
    assert abs(peaks["break"] - k.peak_knee_ext_vel) < abs(peaks["through"] - k.peak_knee_ext_vel)
    assert peaks["break"] == pytest.approx(k.peak_knee_ext_vel, rel=0.05)


def test_no_ball_withholds_plant_foot_offset_only():
    _, f = _run(make_kick(60), ball=False)
    assert f["plant_foot_offset"]["status"] == "not_measured"
    assert f["knee_angle_contact"]["status"] == "measured"


def test_out_of_bound_values_become_not_measured_never_clamped():
    assert violation("trunk_lean_contact", 75.0) is not None
    assert violation("trunk_lean_contact", 10.0) is None
    assert violation("peak_knee_ext_velocity", 16444.0) is not None  # v1's impossible value
    assert all(b.source for b in BOUNDS.values())
    k = make_kick(60)
    poses = k.poses.copy()
    # Lay the shoulders flat ahead of the hips from just before contact: a 90-deg "lean".
    hip_y = poses[:, skeleton.INDEX["right_hip"], 1]
    for s in ("left_shoulder", "right_shoulder"):
        i = skeleton.INDEX[s]
        poses[k.contact_frame - 3 :, i, 0] = (
            poses[k.contact_frame - 3 :, skeleton.INDEX["right_hip"], 0] + 160
        )
        poses[k.contact_frame - 3 :, i, 1] = hip_y[k.contact_frame - 3 :]
    res = analyse(poses, _meta(60, True), ball_detections(k))
    lean = {f["id"]: f for f in res["features"]}["trunk_lean_contact"]
    assert lean["status"] == "not_measured" and "value" not in lean
    assert "sanity bound" in lean["reason"]


def test_a_rear_view_is_refused_with_reasons_and_no_features():
    k = make_kick(60)
    poses = k.poses.copy()
    # Shoulders a torso-length apart: what a camera behind the player sees.
    mid = (
        poses[:, skeleton.INDEX["left_shoulder"], :2]
        + poses[:, skeleton.INDEX["right_shoulder"], :2]
    ) / 2
    poses[:, skeleton.INDEX["left_shoulder"], 0] = mid[:, 0] - 80
    poses[:, skeleton.INDEX["right_shoulder"], 0] = mid[:, 0] + 80
    res = analyse(poses, _meta(60, True), ball_detections(k))
    assert not res["quality"]["accepted"]
    assert any("side-on" in r for r in res["quality"]["reasons"])
    assert res["features"] == []
