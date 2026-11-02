"""The stick figure: drawn from the cleaned signal at contact and backswing, only for an
accepted clip, with unknown points left as None."""

import json
from pathlib import Path

import numpy as np

from kickcoach.figure import frame_size, pose_figure
from kickcoach.pose.skeleton import INDEX, K

ROOT = Path(__file__).resolve().parents[1]


def test_figure_holds_the_two_measured_frames_and_marks_unknown_points():
    res = json.loads((ROOT / "results" / "pipeline" / "c07.json").read_text())
    xy = np.random.default_rng(0).uniform(0, 400, (150, K, 2))
    xy[76, INDEX["left_heel"]] = np.nan
    fig = pose_figure(xy, res, (480, 360))
    assert {n: f["frame"] for n, f in fig["frames"].items()} == {"contact": 76, "max_backswing": 49}
    pts = fig["frames"]["contact"]["points"]
    assert pts["left_heel"] is None
    assert pts["right_knee"] == [round(float(v), 1) for v in xy[76, INDEX["right_knee"]]]
    assert fig["frame_size"] == [480, 360] and fig["kicking_leg"] == "right"


def test_no_figure_for_a_refused_clip():
    res = json.loads((ROOT / "results" / "pipeline" / "c26.json").read_text())
    assert pose_figure(np.zeros((10, K, 2)), res, (480, 360)) is None


def test_frame_size_prefers_the_upright_size():
    stream = {"stream": {"width": 1920, "height": 1080}}
    assert frame_size(stream) == (1920, 1080)
    assert frame_size({**stream, "frame_size": [1080, 1920]}) == (1080, 1920)
