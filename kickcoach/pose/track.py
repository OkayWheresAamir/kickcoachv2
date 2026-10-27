"""Pick the kicker's box in each frame when a detector finds several people.

Start with the largest box (the kicker fills most of a side-on capture), then follow the
box that overlaps the previous choice most. If nothing overlaps enough, the track is lost
and the largest box is taken again.
"""

from __future__ import annotations

import numpy as np

# heuristic: below this IoU with the previous box, treat the track as lost.
MIN_IOU = 0.1


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box ``a`` against boxes ``b`` [N, 4]."""
    b = np.atleast_2d(b)
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])  # noqa: E731
    union = area(a) + area(b) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


class BoxTracker:
    def __init__(self) -> None:
        self.prev: np.ndarray | None = None

    def select(self, boxes: np.ndarray) -> np.ndarray | None:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return None
        if self.prev is not None:
            ov = iou(self.prev, boxes)
            if ov.max() >= MIN_IOU:
                self.prev = boxes[int(ov.argmax())]
                return self.prev
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        self.prev = boxes[int(areas.argmax())]
        return self.prev
