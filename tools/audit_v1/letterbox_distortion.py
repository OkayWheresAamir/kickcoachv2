"""Show how v1's keypoint mapping distorts angles (simulation, no model involved).

v1 letterboxed each frame to 192x192 (tf.image.resize_with_pad) but mapped MoveNet's
normalised output back with ``y_norm * height, x_norm * width``, ignoring the padding.
This simulates known poses through that mapping and through the corrected one.
Run: uv run python tools/audit_v1/letterbox_distortion.py
"""

from __future__ import annotations

import numpy as np

from kickcoach.features.geometry import joint_angle_deg, lean_from_vertical_deg
from kickcoach.pose.letterbox import Letterbox


def v1_to_source(xy_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(xy_norm) * np.array([width, height])


def leg(knee_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hip = np.array([900.0, 300.0])
    knee = hip + [0.0, 250.0]
    t = np.radians(180.0 - knee_deg)
    return hip, knee, knee + 250.0 * np.array([np.sin(t), np.cos(t)])


def main() -> None:
    for width, height, name in [(1920, 1080, "landscape 16:9"), (1080, 1920, "portrait 9:16")]:
        lb = Letterbox(width, height, 192)

        def v1(p, lb=lb, width=width, height=height):
            return v1_to_source(lb.to_square_norm(p), width, height)

        def fixed(p, lb=lb):
            return lb.to_source(lb.to_square_norm(p))

        print(name)
        for true_knee in (90, 120, 150, 170):
            pts = leg(true_knee)
            print(
                f"  knee  true {true_knee:5.1f}  v1 {joint_angle_deg(*map(v1, pts)):6.1f}"
                f"  fixed {joint_angle_deg(*map(fixed, pts)):6.1f}"
            )
        hip = np.array([900.0, 600.0])
        t = np.radians(15.0)
        shoulder = hip + 300.0 * np.array([np.sin(t), -np.cos(t)])
        print(
            f"  trunk true  15.0  v1 {lean_from_vertical_deg(v1(hip), v1(shoulder), 1):6.1f}"
            f"  fixed {lean_from_vertical_deg(fixed(hip), fixed(shoulder), 1):6.1f}"
        )


if __name__ == "__main__":
    main()
