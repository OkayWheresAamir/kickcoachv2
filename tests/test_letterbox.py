import numpy as np
import pytest

from kickcoach.features.geometry import joint_angle_deg
from kickcoach.pose.letterbox import Letterbox


def _v1_to_source(xy_norm, width, height):
    """v1's mapping (pose_engine.scaled_keypoints_from_output): ignores the padding."""
    return np.asarray(xy_norm) * np.array([width, height])


def _leg(knee_angle_deg):
    """Hip, knee, ankle with a vertical thigh and the shank swung forward."""
    hip = np.array([900.0, 300.0])
    knee = hip + [0.0, 250.0]
    t = np.radians(180.0 - knee_angle_deg)
    ankle = knee + 250.0 * np.array([np.sin(t), np.cos(t)])
    return hip, knee, ankle


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (1080, 1920), (640, 640)])
def test_round_trip_is_exact(width, height):
    lb = Letterbox(width, height, 192)
    pts = np.array([[0, 0], [width, height], [width / 3, height * 0.7]])
    np.testing.assert_allclose(lb.to_source(lb.to_square_norm(pts)), pts, atol=1e-9)


def test_padding_is_on_the_short_axis_only():
    np.testing.assert_allclose(Letterbox(1920, 1080, 192).pad, [0.0, 42.0])
    np.testing.assert_allclose(Letterbox(1080, 1920, 192).pad, [42.0, 0.0])


@pytest.mark.parametrize("true_angle", [120.0, 150.0, 170.0])
def test_fixed_mapping_preserves_knee_angle(true_angle):
    lb = Letterbox(1920, 1080, 192)
    recovered = [lb.to_source(lb.to_square_norm(p)) for p in _leg(true_angle)]
    assert joint_angle_deg(*recovered) == pytest.approx(true_angle, abs=1e-6)


def test_v1_mapping_distorts_knee_angle_on_landscape_video():
    """Regression record of the v1 bug: a true 150 deg knee reads as ~134 deg on 16:9."""
    lb = Letterbox(1920, 1080, 192)
    v1 = [_v1_to_source(lb.to_square_norm(p), 1920, 1080) for p in _leg(150.0)]
    assert joint_angle_deg(*v1) == pytest.approx(134.3, abs=0.1)
