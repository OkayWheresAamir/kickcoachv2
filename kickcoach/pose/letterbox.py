"""Map keypoints between a letterboxed (aspect-preserving, centre-padded) square model
input and the source frame.

v1 padded frames to 192x192 but mapped the model's normalised output straight onto the
source frame (``y_norm * height``), ignoring the padding. That compresses the padded axis
by height/width (0.5625 on 16:9 video) and distorts every joint angle; see docs/V1_AUDIT.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Letterbox:
    """Geometry of fitting a ``width`` x ``height`` frame into a ``size`` x ``size`` square."""

    width: int
    height: int
    size: int

    @property
    def scale(self) -> float:
        """Square-input pixels per source pixel."""
        return self.size / max(self.width, self.height)

    @property
    def pad(self) -> np.ndarray:
        """(pad_x, pad_y) in square-input pixels; the image is centred."""
        return np.array(
            [
                (self.size - self.width * self.scale) / 2.0,
                (self.size - self.height * self.scale) / 2.0,
            ]
        )

    def to_source(self, xy_norm: np.ndarray) -> np.ndarray:
        """Coordinates normalised to the square input, shape (..., 2) as (x, y) -> source pixels."""
        xy_norm = np.asarray(xy_norm, dtype=np.float64)
        return (xy_norm * self.size - self.pad) / self.scale

    def to_square_norm(self, xy_src: np.ndarray) -> np.ndarray:
        """Source pixels (..., 2) as (x, y) -> coordinates normalised to the square input."""
        xy_src = np.asarray(xy_src, dtype=np.float64)
        return (xy_src * self.scale + self.pad) / self.size
