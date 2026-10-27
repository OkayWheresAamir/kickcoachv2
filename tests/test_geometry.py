import numpy as np
import pytest

from kickcoach.features.geometry import joint_angle_deg, lean_from_vertical_deg


@pytest.mark.parametrize(
    ("a", "c", "expected"),
    [
        ((1, 0), (1, 0), 0.0),  # folded back on itself
        ((1, 0), (0, 1), 90.0),
        ((1, 0), (-1, 0), 180.0),  # straight
        ((1, 0), (1, 1), 45.0),
        ((2, 0), (-1, np.sqrt(3)), 120.0),  # segment lengths differ
    ],
)
def test_joint_angle_hand_computed(a, c, expected):
    assert joint_angle_deg(a, (0, 0), c) == pytest.approx(expected, abs=1e-9)


def test_joint_angle_degenerate_is_nan():
    assert np.isnan(joint_angle_deg((0, 0), (0, 0), (1, 0)))
    assert np.isnan(joint_angle_deg((1, 0), (0, 0), (0, 0)))


def test_joint_angle_vectorised_over_frames():
    b = np.zeros((3, 2))
    a = np.array([[1, 0], [1, 0], [1, 0]])
    c = np.array([[0, 1], [-1, 0], [0, 0]])
    out = joint_angle_deg(a, b, c)
    assert out.shape == (3,)
    np.testing.assert_allclose(out[:2], [90.0, 180.0])
    assert np.isnan(out[2])


def test_joint_angle_is_scale_and_translation_invariant():
    a, b, c = np.array([3.0, 1.0]), np.array([0.5, 2.0]), np.array([-1.0, 4.0])
    base = joint_angle_deg(a, b, c)
    shifted = joint_angle_deg(a * 7 + 100, b * 7 + 100, c * 7 + 100)
    assert shifted == pytest.approx(base)


def test_lean_upright_is_zero():
    # Shoulder directly above hip (y is down, so "above" is smaller y).
    assert lean_from_vertical_deg((100, 500), (100, 200), facing=1) == pytest.approx(0.0)


@pytest.mark.parametrize("facing", [1, -1])
def test_lean_forward_is_positive_for_both_kick_directions(facing):
    hip = np.array([100.0, 500.0])
    shoulder = hip + np.array([facing * 100.0, -100.0])  # 45 deg toward the kick direction
    assert lean_from_vertical_deg(hip, shoulder, facing) == pytest.approx(45.0)
    leaning_back = hip + np.array([-facing * 100.0, -100.0])
    assert lean_from_vertical_deg(hip, leaning_back, facing) == pytest.approx(-45.0)


def test_lean_degenerate_and_bad_facing():
    assert np.isnan(lean_from_vertical_deg((5, 5), (5, 5), facing=1))
    with pytest.raises(ValueError):
        lean_from_vertical_deg((0, 0), (0, -1), facing=0)
