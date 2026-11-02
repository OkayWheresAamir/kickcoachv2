"""Ball candidates per frame from a COCO-trained detector (handoff D9: "sports ball").

Model: Megvii's YOLOX-m COCO ONNX (models/yolox/yolox_m.onnx, Apache-2.0), the file rtmlib's
own Animal solution uses. rtmlib's YOLOX wrapper does the letterbox and the inference; the
decode is ours, because rtmlib's drops the scores and keeps only boxes scoring >= 0.7, which
a small, blurred ball rarely reaches. Candidates are kept at a low score and the tracker
(kickcoach.events.ball) decides which one is the ball.

Cached per clip as data/cache/ball/<clip>.json: {"frames": [[[x1, y1, x2, y2, score], ...]]}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kickcoach.pose.ort_threads import cap_threads

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolox" / "yolox_m.onnx"
CACHE_DIR = ROOT / "data" / "cache" / "ball"
SPORTS_BALL = 32  # index in rtmlib's YOLOX.COCO_CLASSES (COCO-80 order)
INPUT_SIZE = (640, 640)
STRIDES = (8, 16, 32)
# heuristic: low enough to keep a blurred ball as a candidate; the tracker filters.
MIN_SCORE = 0.05
NMS_IOU = 0.45  # rtmlib's YOLOX default
MAX_PER_FRAME = 5


def decode(raw: np.ndarray, ratio: float, cls: int = SPORTS_BALL) -> np.ndarray:
    """YOLOX head output [N, 85] (no NMS inside) -> [M, 5] xyxy + score in source pixels."""
    grids, strides = [], []
    for s in STRIDES:
        h, w = INPUT_SIZE[0] // s, INPUT_SIZE[1] // s
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
        strides.append(np.full((h * w, 1), s))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    cxcy = (raw[:, :2] + grid) * stride
    wh = np.exp(raw[:, 2:4]) * stride
    score = raw[:, 4] * raw[:, 5 + cls]
    keep = score >= MIN_SCORE
    if not keep.any():
        return np.zeros((0, 5))
    boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], 1)[keep] / ratio
    from rtmlib.tools.object_detection.post_processings import nms

    idx = nms(boxes, score[keep], NMS_IOU)[:MAX_PER_FRAME]
    return np.concatenate([boxes[idx], score[keep][idx, None]], 1)


class BallDetector:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        from rtmlib.tools.object_detection.yolox import YOLOX

        self.model_path = Path(model_path)
        self._yolox = YOLOX(
            str(self.model_path), model_input_size=INPUT_SIZE, backend="onnxruntime"
        )
        cap_threads(self._yolox)

    def detect(self, rgb: np.ndarray) -> np.ndarray:
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # Megvii's ONNX was exported for cv2 BGR
        img, ratio = self._yolox.preprocess(bgr)
        raw = self._yolox.inference(img)[0][0]
        return decode(raw, ratio)


def cache_path(clip_id: str, root: Path = CACHE_DIR) -> Path:
    return root / f"{clip_id}.json"


def save(clip_id: str, frames: list[np.ndarray], info: dict, root: Path = CACHE_DIR) -> Path:
    p = cache_path(clip_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {**info, "frames": [np.round(f, 2).tolist() for f in frames]}
    p.write_text(json.dumps(data))
    return p


def load(clip_id: str, root: Path = CACHE_DIR) -> list[np.ndarray]:
    data = json.loads(cache_path(clip_id, root).read_text())
    return [np.asarray(f, dtype=np.float64).reshape(-1, 5) for f in data["frames"]]
