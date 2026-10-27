"""Pose model adapters. Each maps one clip's RGB frames to canonical [T, K, 3] (x, y, conf)
in source-frame pixels. Model runtimes are imported lazily so the package imports without
them (e.g. in tests or the API).

- movenet:   v1's MoveNet SinglePose Lightning (192x192), whole frame, letterbox fixed.
- rtmpose:   RTMPose-m Halpe26 (body + feet) on a YOLOX-m person box, via rtmlib.
- mediapipe: MediaPipe Pose Landmarker (heavy) in VIDEO mode (tracks across frames).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from kickcoach.pose import skeleton
from kickcoach.pose.letterbox import PixelLetterbox
from kickcoach.pose.track import BoxTracker

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MoveNet:
    name = "movenet"
    SIZE = 192

    def __init__(self, model_path: Path = MODELS_DIR / "movenet" / "v1_3.tflite") -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.model_path = Path(model_path)
        self._it = Interpreter(model_path=str(self.model_path), num_threads=4)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]
        if tuple(self._in["shape"]) != (1, self.SIZE, self.SIZE, 3):
            raise ValueError(f"unexpected MoveNet input {self._in['shape']}")

    def raw(self, img: np.ndarray) -> tuple[np.ndarray, PixelLetterbox]:
        """Model output [17, 3] as (y, x, score) normalised to the padded square."""
        lb = PixelLetterbox.fit(img.shape[1], img.shape[0], self.SIZE)
        x = lb.apply(img).astype(np.float32)[None]  # float32 in [0, 255], as v1 fed it
        self._it.set_tensor(self._in["index"], x)
        self._it.invoke()
        return self._it.get_tensor(self._out["index"])[0, 0].copy(), lb

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        out, lb = self.raw(rgb)
        xy = lb.to_source(out[:, [1, 0]])
        return skeleton.from_named(xy, out[:, 2], skeleton.KEYPOINTS[:17])

    def run_clip(self, frames: Iterable[np.ndarray], times_s: Sequence[float]) -> np.ndarray:
        return np.stack([self.predict(f) for f in frames]).astype(np.float32)

    def info(self) -> dict:
        return {
            "model": "MoveNet SinglePose Lightning (v1's 3.tflite)",
            "input": "whole frame letterboxed to 192x192, RGB float32 0-255",
            "weights": str(self.model_path.relative_to(MODELS_DIR.parent)),
            "weights_sha256": sha256(self.model_path),
        }


class RTMPoseFeet:
    name = "rtmpose"

    def __init__(self, mode: str = "balanced") -> None:
        from rtmlib import BodyWithFeet

        self.mode = mode
        self._model = BodyWithFeet(mode=mode, backend="onnxruntime", device="cpu")

    def run_clip(self, frames: Iterable[np.ndarray], times_s: Sequence[float]) -> np.ndarray:
        out = skeleton.empty(len(times_s))
        tracker = BoxTracker()
        for t, rgb in enumerate(frames):
            bgr = np.ascontiguousarray(rgb[..., ::-1])  # rtmlib's documented input is cv2 BGR
            box = tracker.select(self._model.det_model(bgr))
            if box is None:
                continue
            kps, scores = self._model.pose_model(bgr, bboxes=[box])
            out[t] = skeleton.from_named(kps[0], scores[0], skeleton.HALPE26)
        return out

    def info(self) -> dict:
        from rtmlib import BodyWithFeet

        cfg = BodyWithFeet.MODE[self.mode]
        return {
            "model": f"RTMPose Halpe26 via rtmlib BodyWithFeet mode={self.mode}",
            "detector": cfg["det"],
            "pose": cfg["pose"],
            "detector_onnx_sha256": sha256(self._model.det_model.onnx_model),
            "pose_onnx_sha256": sha256(self._model.pose_model.onnx_model),
            "person_selection": "largest box, then max IoU with previous (kickcoach.pose.track)",
        }


class MediaPipePose:
    name = "mediapipe"

    def __init__(
        self, model_path: Path = MODELS_DIR / "mediapipe" / "pose_landmarker_heavy.task"
    ) -> None:
        self.model_path = Path(model_path)

    def run_clip(self, frames: Iterable[np.ndarray], times_s: Sequence[float]) -> np.ndarray:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        lm_names = [lm.name for lm in vision.PoseLandmark]
        names = tuple(skeleton.MEDIAPIPE_TO_CANONICAL.get(n, "") for n in lm_names)
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        out = skeleton.empty(len(times_s))
        t0 = times_s[0]
        # A fresh landmarker per clip: VIDEO mode needs increasing timestamps and tracks state.
        with vision.PoseLandmarker.create_from_options(opts) as lmk:
            for t, rgb in enumerate(frames):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                res = lmk.detect_for_video(image, int(round((times_s[t] - t0) * 1000)))
                if not res.pose_landmarks:
                    continue
                h, w = rgb.shape[:2]
                lms = res.pose_landmarks[0]
                xy = np.array([[lm.x * w, lm.y * h] for lm in lms])
                conf = np.array(
                    [lm.visibility if lm.visibility is not None else np.nan for lm in lms]
                )
                out[t] = skeleton.from_named(xy, conf, names)
        return out

    def info(self) -> dict:
        return {
            "model": "MediaPipe Pose Landmarker heavy (float16), VIDEO mode, num_poses=1",
            "weights": str(self.model_path.relative_to(MODELS_DIR.parent)),
            "weights_sha256": sha256(self.model_path),
            "conf": "landmark visibility",
            "note": "big_toe is MediaPipe's FOOT_INDEX (tip of the foot); no small-toe point",
        }


MODELS = {"movenet": MoveNet, "rtmpose": RTMPoseFeet, "mediapipe": MediaPipePose}
