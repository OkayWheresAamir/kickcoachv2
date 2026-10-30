"""Run the ball detector on the ingested frames and cache candidates per clip.

uv run python tools/run_ball.py              # all manifest clips not yet cached
uv run python tools/run_ball.py c01 --force  # recompute
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from kickcoach.events import ball_detector
from kickcoach.pose.models import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    with open(ROOT / "data" / "manifest.csv", newline="") as f:
        clips = args.clips or [r["clip_id"] for r in csv.DictReader(f)]
    det = ball_detector.BallDetector()
    info = {
        "model": "YOLOX-m COCO (Megvii 0.1.1rc0 ONNX), class 'sports ball'",
        "weights_sha256": sha256(det.model_path),
        "min_score": ball_detector.MIN_SCORE,
    }
    for cid in clips:
        if ball_detector.cache_path(cid).exists() and not args.force:
            continue
        t0 = time.time()
        paths = sorted((ROOT / "data" / "frames" / cid).glob("*.jpg"))
        out = [det.detect(np.asarray(Image.open(p).convert("RGB"))) for p in paths]
        dt = time.time() - t0
        ball_detector.save(cid, out, {**info, "seconds": round(dt, 1)})
        hit = sum(len(o) > 0 for o in out)
        print(f"{cid}: {len(out)} frames, candidates in {hit}, {len(out) / dt:.1f} frames/s")


if __name__ == "__main__":
    main()
