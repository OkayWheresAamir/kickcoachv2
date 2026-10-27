"""YouTube metadata via yt-dlp, cached on disk so curation never re-queries YouTube.

Only metadata and preview images (thumbnails, storyboards) are read here; video bytes are
fetched by tools/fetch_clips.py, and only for clips in the approved manifest.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import requests
from PIL import Image
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "cache" / "yt_meta"

# YouTube extraction needs a JS runtime; yt-dlp enables only deno by default, and this machine
# has node (yt-dlp >= 2025.11, see its --js-runtimes help).
YDL_BASE = {"quiet": True, "no_warnings": True, "js_runtimes": {"node": {}}}


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def load_meta(video_id: str) -> dict:
    path = META_DIR / f"{video_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    with YoutubeDL({**YDL_BASE, "skip_download": True}) as ydl:
        info = ydl.sanitize_info(ydl.extract_info(watch_url(video_id), download=False))
    META_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return info


def video_formats(meta: dict) -> list[dict]:
    return [f for f in meta.get("formats", []) if f.get("vcodec") not in (None, "none")]


def storyboard_tiles(meta: dict) -> list[tuple[float, Image.Image]]:
    """(time_s, tile) pairs from the highest-resolution storyboard; [] if there is none."""
    boards = [f for f in meta.get("formats", []) if "storyboard" in (f.get("format_note") or "")]
    if not boards:
        return []
    sb = max(boards, key=lambda f: f.get("width") or 0)
    w, h, rows, cols = sb["width"], sb["height"], sb["rows"], sb["columns"]
    step = 1.0 / sb["fps"]
    tiles: list[tuple[float, Image.Image]] = []
    t = 0.0
    for frag in sb["fragments"]:
        resp = requests.get(frag["url"], timeout=30)
        resp.raise_for_status()
        mosaic = Image.open(io.BytesIO(resp.content)).convert("RGB")
        for k in range(rows * cols):
            if t > meta["duration"]:
                break
            r, c = divmod(k, cols)
            tile = mosaic.crop((c * w, r * h, (c + 1) * w, (r + 1) * h))
            tiles.append((t, tile))
            t += step
    return tiles


def thumbnail(meta: dict) -> Image.Image:
    resp = requests.get(meta["thumbnail"], timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")
