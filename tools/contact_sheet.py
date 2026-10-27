"""Build the clip-approval contact sheet from data/candidates.csv.

Each proposed clip shows YouTube storyboard frames from inside its window, so the camera view
and the kick can be judged without downloading any video. Output (gitignored, because it
contains frames of other people's videos): data/cache/contact_sheet/index.html

    uv run python tools/contact_sheet.py
"""

from __future__ import annotations

import base64
import csv
import html
import io

from PIL import Image
from ytmeta import ROOT, load_meta, storyboard_tiles, thumbnail, video_formats

OUT = ROOT / "data" / "cache" / "contact_sheet" / "index.html"
TILES_PER_CLIP = 8
MIN_TILE_WIDTH = 100  # smaller storyboards are unreadable; fall back to the thumbnail

GROUP_TITLES = {
    "side_on": "Proposed: side-on kicks (the evaluation set)",
    "view_negative": "Proposed: wrong-view kicks (quality-gate negatives for E8, no keypoints)",
    "excluded": "Excluded (shown so you can overrule)",
}


def _b64(img: Image.Image, width: int) -> str:
    img = img.copy()
    img.thumbnail((width, width))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _window_tiles(meta: dict, start: float, end: float) -> list[tuple[str, Image.Image]]:
    tiles = storyboard_tiles(meta)
    if tiles and tiles[0][1].width >= MIN_TILE_WIDTH:
        inside = [(t, im) for t, im in tiles if start <= t <= end]
        if len(inside) > TILES_PER_CLIP:
            idx = [
                round(i * (len(inside) - 1) / (TILES_PER_CLIP - 1)) for i in range(TILES_PER_CLIP)
            ]
            inside = [inside[i] for i in idx]
        return [(f"{t:.1f}s", im) for t, im in inside]
    return [("thumbnail", thumbnail(meta))]


def _card(row: dict, meta: dict) -> str:
    vf = video_formats(meta)
    fps = sorted({f["fps"] for f in vf if f.get("fps")})
    max_h = max((f.get("height") or 0) for f in vf) if vf else 0
    excluded = row["group"] == "excluded"
    start = float(row["start_s"]) if row["start_s"] else 0.0
    end = float(row["end_s"]) if row["end_s"] else float(meta["duration"])
    tiles = [("thumbnail", thumbnail(meta))] if excluded else _window_tiles(meta, start, end)
    width = 200 if excluded else 240
    imgs = "".join(
        f'<figure><img src="data:image/jpeg;base64,{_b64(im, width)}"><figcaption>{lbl}'
        "</figcaption></figure>"
        for lbl, im in tiles
    )
    link = f"https://youtu.be/{meta['id']}?t={int(start)}"
    window = "" if excluded else f" · window <b>{start:g}–{end:g} s</b>"
    cap = (
        f" · capture fps stated: <b>{row['capture_fps_stated']}</b>"
        if row["capture_fps_stated"]
        else ""
    )
    return f"""
<section class="card {row["group"]}">
  <h3>{row["cand_id"]} · <a href="{link}" target="_blank">{html.escape(meta["title"])}</a></h3>
  <p class="meta">{html.escape(meta.get("channel") or "")} · {row["source"]} ·
    {meta["duration"]} s · playback fps {fps} · up to {max_h}p{window}{cap} ·
    level guess: {row["level_guess"]}</p>
  <p class="notes">{html.escape(row["notes"])}</p>
  <div class="strip">{imgs}</div>
</section>"""


def main() -> None:
    rows = list(csv.DictReader((ROOT / "data" / "candidates.csv").open()))
    parts = []
    for group, title in GROUP_TITLES.items():
        group_rows = [r for r in rows if r["group"] == group]
        parts.append(f"<h2>{title} — {len(group_rows)}</h2>")
        for r in group_rows:
            print(r["cand_id"], r["video_id"], flush=True)
            parts.append(_card(r, load_meta(r["video_id"])))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        """<!doctype html><html><head><meta charset="utf-8"><title>KickCoach clip candidates</title>
<style>
body{font:14px/1.4 -apple-system,system-ui,sans-serif;margin:16px;background:#fafafa;color:#111}
h2{margin-top:32px;border-bottom:2px solid #333}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 12px;margin:10px 0}
.card.view_negative{border-left:6px solid #d98c00}.card.excluded{opacity:.8}
.card.side_on{border-left:6px solid #2a7}
h3{margin:2px 0;font-size:15px}.meta{color:#555;margin:2px 0}.notes{margin:4px 0}
.strip{display:flex;gap:4px;overflow-x:auto}figure{margin:0}
figcaption{font-size:11px;color:#666;text-align:center}img{display:block;max-height:180px}
</style></head><body>
<h1>KickCoach v2 — clip candidates for approval</h1>
<p>Frames are YouTube storyboard previews (about 1 per second) from inside each proposed window.
Click a title to watch from the window start. Windows are trimmed tighter after download.</p>
"""
        + "\n".join(parts)
        + "</body></html>"
    )
    print(OUT)


if __name__ == "__main__":
    main()
