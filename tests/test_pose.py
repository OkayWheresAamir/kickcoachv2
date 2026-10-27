import numpy as np
import pytest

from kickcoach.pose import cache, skeleton
from kickcoach.pose.letterbox import PixelLetterbox
from kickcoach.pose.track import BoxTracker, iou


@pytest.mark.parametrize(("w", "h"), [(640, 352), (1920, 1080), (480, 360), (360, 640)])
def test_pixel_letterbox_round_trip(w, h):
    lb = PixelLetterbox.fit(w, h, 192)
    pts = np.array([[0.0, 0.0], [w, h], [w * 0.31, h * 0.77]])
    np.testing.assert_allclose(lb.to_source(lb.to_square_norm(pts)), pts, atol=1e-9)


def test_pixel_letterbox_uses_the_rounded_resize():
    lb = PixelLetterbox.fit(640, 352, 192)
    assert (lb.new_w, lb.new_h, lb.pad_x, lb.pad_y) == (192, 106, 0, 43)


def test_pixel_letterbox_image_lands_where_the_mapping_says():
    w, h = 640, 352
    img = np.zeros((h, w, 3), np.uint8)
    img[200:210, 500:510] = 255  # a bright square centred at (505, 205)
    lb = PixelLetterbox.fit(w, h, 192)
    sq = lb.apply(img)
    ys, xs = np.nonzero(sq[..., 0] > 60)
    centre_norm = np.array([xs.mean() + 0.5, ys.mean() + 0.5]) / 192
    np.testing.assert_allclose(lb.to_source(centre_norm), [505, 205], atol=2.0)


def test_from_named_maps_by_name_and_marks_missing():
    names = ("left_knee", "head", "right_heel")
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = skeleton.from_named(xy, np.array([0.9, 0.8, 0.7]), names)
    assert out.shape == (skeleton.K, 3)
    np.testing.assert_allclose(out[skeleton.INDEX["left_knee"]], [1, 2, 0.9], rtol=1e-6)
    np.testing.assert_allclose(out[skeleton.INDEX["right_heel"]], [5, 6, 0.7], rtol=1e-6)
    assert np.isnan(out[skeleton.INDEX["nose"], 0]) and out[skeleton.INDEX["nose"], 2] == 0


def test_halpe26_and_mediapipe_cover_all_canonical_body_points():
    assert set(skeleton.KEYPOINTS) <= set(skeleton.HALPE26)
    mp_targets = set(skeleton.MEDIAPIPE_TO_CANONICAL.values())
    assert set(skeleton.KEYPOINTS) - mp_targets == {"left_small_toe", "right_small_toe"}


def test_tracker_starts_on_largest_then_follows_overlap():
    big = [0, 0, 100, 200]
    small = [300, 0, 340, 60]
    tr = BoxTracker()
    np.testing.assert_allclose(tr.select([small, big]), big)
    # The kicker moves a little and a bystander walks in with a larger box elsewhere.
    moved = [10, 0, 110, 200]
    bystander = [500, 0, 700, 400]
    np.testing.assert_allclose(tr.select([bystander, moved]), moved)
    assert tr.select([]) is None


def test_iou():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    np.testing.assert_allclose(
        iou(a, np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])), [1.0, 1 / 3, 0.0]
    )


def test_cache_round_trip_and_shape_check(tmp_path):
    poses = skeleton.empty(4)
    cache.save("m", "c", poses, {"model": "x"}, root=tmp_path)
    assert cache.load("m", "c", root=tmp_path).shape == (4, skeleton.K, 3)
    with pytest.raises(ValueError):
        cache.save("m", "bad", np.zeros((4, 17, 3)), {}, root=tmp_path)
