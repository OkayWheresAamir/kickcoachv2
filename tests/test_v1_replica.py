import numpy as np
import pytest

from eval import v1_replica
from kickcoach.pose import skeleton
from kickcoach.pose.letterbox import Letterbox

FPS = 30.0


def test_v1_mapping_squashes_the_padded_axis_toward_the_centre():
    # V1_AUDIT 2.2: on 16:9 video v1 compresses vertical offsets from the centre by h/w.
    lb = Letterbox(1920, 1080, 192)
    src = np.array([[960.0, 540.0], [960.0, 0.0], [100.0, 900.0]])
    norm = lb.to_square_norm(src)
    out = np.column_stack([norm[:, 1], norm[:, 0], np.ones(3)])  # MoveNet's (y, x, score)
    v1 = v1_replica.v1_pixels(out, 1920, 1080)
    np.testing.assert_allclose(v1[:, 0], src[:, 0])  # the long axis has no padding
    np.testing.assert_allclose(v1[:, 1] - 540, (src[:, 1] - 540) * 1080 / 1920)


def test_ema_starts_at_the_first_value_and_holds_through_gaps():
    out = v1_replica.ema(np.array([np.nan, 10.0, np.nan, 20.0]), alpha=0.5)
    np.testing.assert_allclose(out, [np.nan, 10.0, 10.0, 15.0])


def _poses(n: int = 10) -> np.ndarray:
    """Both legs straight and still, facing image right; every score 0.9."""
    p = skeleton.empty(n)
    pts = {
        "left_shoulder": (100, 0),
        "right_shoulder": (102, 0),
        "left_hip": (100, 100),
        "right_hip": (102, 100),
        "left_knee": (100, 200),
        "right_knee": (102, 200),
        "left_ankle": (100, 300),
        "right_ankle": (102, 300),
    }
    for name, (x, y) in pts.items():
        p[:, skeleton.INDEX[name]] = (x, y, 0.9)
    return p


def test_kick_is_the_fastest_ankle_and_reports_v1s_numbers():
    p = _poses()
    p[5, skeleton.INDEX["left_ankle"], 0] += 40  # one 40 px jump on frame 5
    a = v1_replica.kick(p, FPS)
    assert (a.frame, a.leg) == (5, "left")
    assert a.ankle_speed == pytest.approx(40 * FPS)
    assert a.knee_vel < 0  # the jump bends the knee
    assert a.knee_angle > 170  # EMA, so most of the bend is smoothed away


def test_ties_go_to_the_right_leg_as_in_v1():
    p = _poses()
    for leg in ("left", "right"):
        p[4, skeleton.INDEX[f"{leg}_ankle"], 0] += 10
    assert v1_replica.kick(p, FPS).leg == "right"


def test_knee_velocity_spans_gaps_as_v1_did():
    # V1_AUDIT 2.4: the change since the last valid angle is divided by one frame's time.
    p = _poses(5)
    p[1:3, skeleton.INDEX["right_knee"], 2] = 0.1  # below v1's 0.2: no angle on frames 1-2
    p[3, skeleton.INDEX["right_ankle"], 0] += 50
    s = v1_replica.series(p, FPS)
    expected = (s.knee_raw["right"][3] - s.knee_raw["right"][0]) * FPS
    assert s.knee_vel["right"][3] == pytest.approx(expected)
    assert np.isnan(s.knee_vel["right"][1:3]).all()


def test_trunk_sign_follows_the_image_not_the_kick():
    p = _poses()
    for side in ("left", "right"):
        p[:, skeleton.INDEX[f"{side}_shoulder"], 0] += 30  # shoulders toward image right
    assert v1_replica.series(p, FPS).trunk_ema[-1] > 0
