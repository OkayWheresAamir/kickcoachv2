"""Pose outputs cached per model and clip: data/cache/pose/<model>/<clip>.npy + .json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kickcoach.pose import skeleton

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "pose"


def path(model: str, clip_id: str, root: Path = CACHE_DIR) -> Path:
    return root / model / f"{clip_id}.npy"


def save(model: str, clip_id: str, poses: np.ndarray, info: dict, root: Path = CACHE_DIR) -> Path:
    check_shape(poses)
    p = path(model, clip_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, poses.astype(np.float32))
    p.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return p


def load(model: str, clip_id: str, root: Path = CACHE_DIR) -> np.ndarray:
    poses = np.load(path(model, clip_id, root))
    check_shape(poses)
    return poses


def check_shape(poses: np.ndarray) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (skeleton.K, 3):
        raise ValueError(f"pose array must be [T, {skeleton.K}, 3], got {poses.shape}")
