"""The canonical skeleton every pose adapter maps into: COCO-17 body + 3 points per foot.

Order of the first 17 is COCO's (the same in MoveNet and RTMPose Halpe26; checked against
rtmlib/visualization/skeleton/{coco17,halpe26}.py). Foot points follow COCO-WholeBody's
naming. A model that lacks a point reports (nan, nan, 0) for it.

Confidence is each model's own score and is NOT comparable across models: MoveNet gives a
per-keypoint score, RTMPose a SimCC peak value, MediaPipe a landmark visibility.
"""

from __future__ import annotations

import numpy as np

KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)  # fmt: skip
K = len(KEYPOINTS)
INDEX = {name: i for i, name in enumerate(KEYPOINTS)}

HALPE26: tuple[str, ...] = KEYPOINTS[:17] + (
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)  # fmt: skip

# MediaPipe PoseLandmark names (mediapipe/tasks/python/vision/pose_landmarker.py) -> ours.
# FOOT_INDEX is the tip of the foot, the nearest MediaPipe has to a big-toe point.
MEDIAPIPE_TO_CANONICAL = {
    "NOSE": "nose", "LEFT_EYE": "left_eye", "RIGHT_EYE": "right_eye",
    "LEFT_EAR": "left_ear", "RIGHT_EAR": "right_ear",
    "LEFT_SHOULDER": "left_shoulder", "RIGHT_SHOULDER": "right_shoulder",
    "LEFT_ELBOW": "left_elbow", "RIGHT_ELBOW": "right_elbow",
    "LEFT_WRIST": "left_wrist", "RIGHT_WRIST": "right_wrist",
    "LEFT_HIP": "left_hip", "RIGHT_HIP": "right_hip",
    "LEFT_KNEE": "left_knee", "RIGHT_KNEE": "right_knee",
    "LEFT_ANKLE": "left_ankle", "RIGHT_ANKLE": "right_ankle",
    "LEFT_HEEL": "left_heel", "RIGHT_HEEL": "right_heel",
    "LEFT_FOOT_INDEX": "left_big_toe", "RIGHT_FOOT_INDEX": "right_big_toe",
}  # fmt: skip


def empty(n_frames: int) -> np.ndarray:
    """[T, K, 3] filled with 'not detected': (nan, nan, 0)."""
    out = np.full((n_frames, K, 3), np.nan, dtype=np.float32)
    out[..., 2] = 0.0
    return out


def from_named(xy: np.ndarray, conf: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """One frame of a model's keypoints in its own order -> canonical [K, 3]."""
    out = empty(1)[0]
    for j, name in enumerate(names):
        i = INDEX.get(name)
        if i is not None:
            out[i] = (xy[j, 0], xy[j, 1], conf[j])
    return out
