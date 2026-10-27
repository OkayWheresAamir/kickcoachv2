"""Run pose models on the ingested frames and cache [T, K, 3] arrays.

uv run python tools/run_pose.py                     # all models, all manifest clips
uv run python tools/run_pose.py --models rtmpose c01 c02
uv run python tools/run_pose.py --force             # recompute cached outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import certifi
import numpy as np
from PIL import Image

from kickcoach.pose import cache
from kickcoach.pose.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
# rtmlib fetches its ONNX weights with urllib; python.org's macOS Python ships without a CA
# bundle, so point it at certifi's (already a dependency via requests).
os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def frames(clip_id: str):
    for p in sorted((ROOT / "data" / "frames" / clip_id).glob("*.jpg")):
        yield np.asarray(Image.open(p).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = list(csv.DictReader((ROOT / "data" / "manifest.csv").open()))
    clips = args.clips or [r["clip_id"] for r in manifest]
    for name in args.models.split(","):
        todo = [c for c in clips if args.force or not cache.path(name, c).exists()]
        if not todo:
            print(f"{name}: all cached")
            continue
        model = MODELS[name]()
        info = model.info()
        for c in todo:
            meta = json.loads((ROOT / "data" / "frames" / c / "meta.json").read_text())
            t0 = time.perf_counter()
            poses = model.run_clip(frames(c), meta["frame_times_s"])
            secs = time.perf_counter() - t0
            detected = np.isfinite(poses[..., 0]).mean()
            cache.save(name, c, poses, {**info, "clip_id": c, "seconds": round(secs, 1)})
            print(
                f"{name} {c}: {poses.shape} {len(poses) / secs:.1f} frames/s, "
                f"{detected:.0%} keypoints present",
                flush=True,
            )


if __name__ == "__main__":
    main()
