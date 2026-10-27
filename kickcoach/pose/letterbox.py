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


@dataclass(frozen=True)
class PixelLetterbox:
    """The letterbox an actual image resize produces: whole-pixel resized size and offsets.

    ``Letterbox`` is the ideal geometry; a real resize rounds the scaled size to integers
    (e.g. 640x352 -> 192x106, not 192x105.6), so the inverse must use these exact numbers.
    """

    width: int
    height: int
    size: int
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int

    @classmethod
    def fit(cls, width: int, height: int, size: int) -> PixelLetterbox:
        scale = size / max(width, height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        return cls(width, height, size, new_w, new_h, (size - new_w) // 2, (size - new_h) // 2)

    def to_source(self, xy_norm: np.ndarray) -> np.ndarray:
        """Coordinates normalised to the square input, (..., 2) as (x, y) -> source pixels."""
        xy_sq = np.asarray(xy_norm, dtype=np.float64) * self.size
        return (xy_sq - [self.pad_x, self.pad_y]) * [
            self.width / self.new_w,
            self.height / self.new_h,
        ]

    def to_square_norm(self, xy_src: np.ndarray) -> np.ndarray:
        xy_src = np.asarray(xy_src, dtype=np.float64)
        sq = xy_src * [self.new_w / self.width, self.new_h / self.height] + [self.pad_x, self.pad_y]
        return sq / self.size

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Resize ``img`` [H, W, C] into a zero-padded ``size`` x ``size`` square."""
        import cv2  # only needed when running a model

        if img.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"expected {self.width}x{self.height}, got {img.shape[1]}x{img.shape[0]}"
            )
        out = np.zeros((self.size, self.size, img.shape[2]), dtype=img.dtype)
        resized = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_AREA)
        out[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w] = resized
        return out
