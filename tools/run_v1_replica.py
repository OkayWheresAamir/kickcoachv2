"""Cache v1's keypoints for every clip, as the hackathon version produced them: MoveNet
Lightning (v1's own 3.tflite) fed BGR frames (V1_AUDIT 2.9), with the output mapped back as
(x_norm * width, y_norm * height), ignoring the letterbox padding (V1_AUDIT 2.2).

The result goes to data/cache/pose/movenet_v1/<clip>.npy in the canonical [T, K, 3] layout,
so E2, E3 and E8 can replay v1 (eval/v1_replica.py) without TensorFlow. One known difference:
v1 resized with tf.image.resize_with_pad (bilinear), this uses the adapter's cv2 INTER_AREA.

    uv run python tools/run_v1_replica.py [clip_id ...]
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.v1_replica import CACHE_MODEL, v1_pixels  # noqa: E402
from kickcoach.pose import cache, skeleton  # noqa: E402
from kickcoach.pose.models import MoveNet  # noqa: E402


def main(clips: list[str]) -> None:
    if not clips:
        with open(ROOT / "data" / "manifest.csv", newline="") as f:
            clips = [r["clip_id"] for r in csv.DictReader(f)]
    model = MoveNet()
    for cid in clips:
        t0 = time.time()
        rows = []
        for p in sorted((ROOT / "data" / "frames" / cid).glob("*.jpg")):
            rgb = np.asarray(Image.open(p).convert("RGB"))
            h, w = rgb.shape[:2]
            out, _ = model.raw(np.ascontiguousarray(rgb[..., ::-1]))  # BGR, as v1 fed it
            rows.append(
                skeleton.from_named(v1_pixels(out, w, h), out[:, 2], skeleton.KEYPOINTS[:17])
            )
        poses = np.stack(rows).astype(np.float32)
        info = {
            **model.info(),
            "input": "BGR (v1's OpenCV order), letterboxed to 192x192",
            "mapping": "v1: (x_norm * width, y_norm * height), padding ignored (V1_AUDIT 2.2)",
        }
        cache.save(CACHE_MODEL, cid, poses, info)
        print(f"{cid}: {len(poses)} frames in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
