"""Download the source video of every clip in data/manifest.csv into data/raw/src/ (gitignored).

Video only (no audio), H.264 preferred, at most 1080p. There is no ffmpeg CLI on the dev
machine, so yt-dlp can't cut sections; trimming happens in tools/ingest_clips.py with PyAV.

    uv run python tools/fetch_clips.py
"""

from __future__ import annotations

import csv
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from ytmeta import ROOT, YDL_BASE

SRC_DIR = ROOT / "data" / "raw" / "src"
FORMAT = "bv*[vcodec^=avc1][height<=1080]/bv*[height<=1080]"


def video_id(url: str) -> str:
    return parse_qs(urlparse(url).query)["v"][0]


def source_path(vid: str):
    hits = [p for p in SRC_DIR.glob(f"{vid}.*") if p.suffix != ".part"]
    return hits[0] if hits else None


def main() -> None:
    rows = list(csv.DictReader((ROOT / "data" / "manifest.csv").open()))
    ids = sorted({video_id(r["url"]) for r in rows})
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        **YDL_BASE,
        "format": FORMAT,
        "outtmpl": str(SRC_DIR / "%(id)s.%(ext)s"),
        "noprogress": True,
    }
    with YoutubeDL(opts) as ydl:
        for vid in ids:
            if source_path(vid):
                print(f"{vid}: cached")
                continue
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
            print(
                f"{vid}: {info.get('format_id')} {info.get('width')}x{info.get('height')} "
                f"{info.get('fps')} fps {info.get('vcodec')}"
            )
    missing = [v for v in ids if not source_path(v)]
    print(f"{len(ids) - len(missing)}/{len(ids)} sources present; missing: {missing}")


if __name__ == "__main__":
    main()
