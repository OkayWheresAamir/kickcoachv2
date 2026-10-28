"""Local browser tool for labelling kick events and keypoints (Phase 2, DEC-013).

    uv run python tools/label_tool.py               # all manifest clips
    uv run python tools/label_tool.py c10 c01       # only these clips
    uv run python tools/label_tool.py --repeat      # blind relabel of 10 frames (>= 48 h later)
    uv run python tools/label_tool.py --report      # QA flags + timing, no server

Serves tools/label_ui/ on http://127.0.0.1:<port> (local only). Frames come from
data/frames/<clip>/, the keypoint pre-fill from data/cache/pose/rtmpose/<clip>.npy, and
labels are written to data/labels/. The definitions are in docs/LABELLING_GUIDE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from kickcoach.labels import qa, schema, timing
from kickcoach.pose import cache

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent / "label_ui"
STATIC = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}
CLIP_RE = re.compile(r"^c\d{2}$")
MAX_BODY = 5_000_000


class App:
    """Everything the HTTP handler needs; paths are injectable so tests can use a tmp dir."""

    def __init__(
        self,
        clips: list[str],
        manifest: dict[str, dict],
        frames_dir: Path,
        pose_dir: Path,
        labels_dir: Path,
        repeat: dict | None = None,
    ):
        self.clips = clips
        self.manifest = manifest
        self.frames_dir = frames_dir
        self.pose_dir = pose_dir
        self.labels_dir = labels_dir
        self.repeat = repeat  # a repeat session dict when running --repeat, else None

    def info(self, cid: str) -> dict:
        meta = json.loads((self.frames_dir / cid / "meta.json").read_text())
        row = self.manifest.get(cid, {})
        return {
            "clip_id": cid,
            "n_frames": meta["n_frames"],
            "width": meta["stream"]["width"],
            "height": meta["stream"]["height"],
            "purpose": row.get("purpose", "eval"),
            "split": row.get("split", ""),
        }

    def label(self, cid: str) -> dict:
        i = self.info(cid)
        return schema.load(cid, self.labels_dir) or schema.new_label(
            cid, i["n_frames"], i["width"], i["height"], i["purpose"]
        )

    def prefill(self, cid: str) -> list[list]:
        poses = cache.load(schema.PREFILL_MODEL, cid, self.pose_dir)
        return [[schema.prefill_points(row)[n] for n in schema.ALL_POINTS] for row in poses]

    def state(self) -> dict:
        clips = []
        if self.repeat is None:
            for cid in self.clips:
                lab = self.label(cid)
                clips.append({**self.info(cid), "status": schema.status(lab)})
        return {
            "mode": "label" if self.repeat is None else "repeat",
            "clips": clips,
            "points": list(schema.ALL_POINTS),
        }

    def clip_payload(self, cid: str) -> dict:
        lab = self.label(cid)
        return {
            **self.info(cid),
            "prefill": self.prefill(cid),
            "label": lab,
            "qa": qa.flags(lab),
            "status": schema.status(lab),
        }

    def save_label(self, cid: str, lab: dict) -> dict:
        i = self.info(cid)
        # The tool owns these fields; the page can't change which clip or image a file is for.
        lab.update(
            clip_id=cid,
            n_frames=i["n_frames"],
            image_size=[i["width"], i["height"]],
            purpose=i["purpose"],
            schema_version=schema.SCHEMA_VERSION,
            prefill_model=schema.PREFILL_MODEL,
        )
        schema.save(lab, self.labels_dir)
        return {"ok": True, "qa": qa.flags(lab), "status": schema.status(lab)}

    def repeat_payload(self) -> dict:
        tasks = []
        for t in self.repeat["tasks"]:
            row = cache.load(schema.PREFILL_MODEL, t["clip_id"], self.pose_dir)[t["frame"]]
            tasks.append(
                {
                    **t,
                    "n_frames": self.info(t["clip_id"])["n_frames"],
                    "prefill_row": [schema.prefill_points(row)[n] for n in schema.ALL_POINTS],
                }
            )
        return {**self.repeat, "tasks": tasks}

    def save_repeat(self, session: dict) -> dict:
        # Only the human-entered parts of each task are taken from the page.
        for mine, theirs in zip(self.repeat["tasks"], session["tasks"], strict=True):
            for k in ("points", "prefill", "confirmed", "confirmed_at", "active_s", "wall_s"):
                mine[k] = theirs.get(k, mine[k])
        schema.save_repeat(self.repeat, self.labels_dir)
        return {"ok": True}


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache_s: int = 0) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", f"max-age={cache_s}" if cache_s else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _clip(self, cid: str) -> bool:
            if cid not in app.clips:
                self._json({"error": f"unknown clip {cid}"}, 404)
                return False
            if app.repeat is not None:
                # Blind relabel: never send the first labels to the page.
                self._json({"error": "clip labels are hidden in --repeat mode"}, 403)
                return False
            return True

        def do_GET(self) -> None:
            parts = urlparse(self.path).path.strip("/").split("/")
            try:
                if parts == [""]:
                    self._send(200, (UI_DIR / "index.html").read_bytes(), STATIC["index.html"])
                elif len(parts) == 2 and parts[0] == "static" and parts[1] in STATIC:
                    self._send(200, (UI_DIR / parts[1]).read_bytes(), STATIC[parts[1]])
                elif parts == ["api", "state"]:
                    self._json(app.state())
                elif len(parts) == 3 and parts[:2] == ["api", "clip"]:
                    if self._clip(parts[2]):
                        self._json(app.clip_payload(parts[2]))
                elif parts == ["api", "repeat"] and app.repeat is not None:
                    self._json(app.repeat_payload())
                elif (
                    len(parts) == 3
                    and parts[0] == "frames"
                    and CLIP_RE.match(parts[1])
                    and re.fullmatch(r"\d+\.jpg", parts[2])
                ):
                    allowed = (
                        app.clips
                        if app.repeat is None
                        else {t["clip_id"] for t in app.repeat["tasks"]}
                    )
                    p = app.frames_dir / parts[1] / f"{int(parts[2][:-4]):06d}.jpg"
                    if parts[1] in allowed and p.exists():
                        self._send(200, p.read_bytes(), "image/jpeg", cache_s=3600)
                    else:
                        self._json({"error": "no such frame"}, 404)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # show the error in the page instead of a dropped socket
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_PUT(self) -> None:
            parts = urlparse(self.path).path.strip("/").split("/")
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                self._json({"error": "body too large"}, 413)
                return
            try:
                body = json.loads(self.rfile.read(n))
                if len(parts) == 3 and parts[:2] == ["api", "clip"]:
                    if self._clip(parts[2]):
                        self._json(app.save_label(parts[2], body))
                elif parts == ["api", "repeat"] and app.repeat is not None:
                    self._json(app.save_repeat(body))
                else:
                    self._json({"error": "not found"}, 404)
            except ValueError as e:  # includes schema validation errors and bad JSON
                self._json({"error": str(e)}, 400)

    return Handler


def read_manifest() -> dict[str, dict]:
    return {r["clip_id"]: r for r in csv.DictReader((ROOT / "data" / "manifest.csv").open())}


def start_repeat(
    labels_dir: Path, frames_dir: Path, n: int, seed: int, min_age_hours: float
) -> dict:
    """Resume the newest unfinished repeat session, or sample a new one."""
    rdir = schema.repeat_dir(labels_dir)
    for p in sorted(rdir.glob("r*.json"), reverse=True):
        s = json.loads(p.read_text())
        if not all(t["confirmed"] for t in s["tasks"]):
            print(f"resuming {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
            return s
    tasks = schema.sample_repeat_tasks(schema.load_all(labels_dir), n, seed, min_age_hours)
    if not tasks:
        sys.exit(f"no confirmed frames at least {min_age_hours} h old to relabel")
    for t in tasks:
        st = json.loads((frames_dir / t["clip_id"] / "meta.json").read_text())["stream"]
        t["image_size"] = [st["width"], st["height"]]
    session = schema.new_repeat_session(datetime.now().strftime("r%Y%m%d-%H%M"), seed, tasks)
    schema.save_repeat(session, labels_dir)
    print(f"new repeat session {session['session_id']}: {len(tasks)} frames")
    return session


def report(labels_dir: Path, out_dir: Path) -> None:
    labels = schema.load_all(labels_dir)
    if not labels:
        print("no label files yet")
        return
    for cid, lab in labels.items():
        st = schema.status(lab)
        print(
            f"\n{cid} ({lab['purpose']}): events {'done' if st['events_done'] else 'TODO'}, "
            f"frames {st['frames_confirmed']}/{st['frames_needed']} confirmed"
        )
        for f in qa.flags(lab):
            where = f" [frame {f['frame']}]" if f["frame"] is not None else ""
            print(f"  {f['severity']:5} {f['code']}{where}: {f['message']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("frames", timing.frame_rows(labels)), ("clips", timing.clip_rows(labels))):
        with (out_dir / f"timing_{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = timing.summary(labels)
    print("\nTiming (active seconds; idle gaps excluded):")
    if s is None:
        print("  not enough confirmed frames yet")
        return
    print(
        f"  {s['n_frames']} frames: median {s['frame_s_median']:.0f} s, max "
        f"{s['frame_s_max']:.0f} s; events on {s['n_clips']} clips: median "
        f"{s['events_s_median']:.0f} s, max {s['events_s_max']:.0f} s"
    )
    n_eval = sum(r["purpose"] == "eval" for r in read_manifest().values())
    done = sum(
        lab["purpose"] == "eval" and schema.status(lab)["complete"] for lab in labels.values()
    )
    print(f"  projections at these medians ({done} of {n_eval} eval clips already done; the 3")
    print("  gate-only clips need only View OK and are not included):")
    for n_kp, per in ((n_eval, 3), (n_eval, 2), (15, 2)):
        total = timing.projection_minutes(
            s["events_s_median"], s["frame_s_median"], n_eval, n_kp, per
        )
        left = timing.projection_minutes(
            s["events_s_median"], s["frame_s_median"], n_eval - done, n_kp - done, per
        )
        print(
            f"    events on all {n_eval} + keypoints on {n_kp} clips x {per} frames: "
            f"{total:.0f} min in total, {left:.0f} min still to do"
        )
    print(f"  wrote {out_dir / 'timing_frames.csv'} and timing_clips.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("clips", nargs="*", help="clip ids (default: every manifest clip)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--repeat", action="store_true", help="blind relabel of sampled frames")
    ap.add_argument("--n-repeat", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--min-age-hours", type=float, default=48.0)
    ap.add_argument("--report", action="store_true", help="print QA flags and timing, then exit")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--labels-dir", type=Path, default=schema.LABELS_DIR, help="for testing")
    args = ap.parse_args()

    frames_dir, labels_dir = ROOT / "data" / "frames", args.labels_dir
    if args.report:
        report(labels_dir, ROOT / "results" / "labelling")
        return
    manifest = read_manifest()
    clips = args.clips or list(manifest)
    bad = [c for c in clips if c not in manifest or not (frames_dir / c / "meta.json").exists()]
    if bad:
        sys.exit(f"unknown clip(s) or missing frames: {bad} (run tools/ingest_clips.py)")
    missing = [c for c in clips if not cache.path(schema.PREFILL_MODEL, c).exists()]
    if missing:
        sys.exit(
            f"no {schema.PREFILL_MODEL} pre-fill for {missing} (run tools/run_pose.py "
            f"--models {schema.PREFILL_MODEL} {' '.join(missing)})"
        )
    repeat = None
    if args.repeat:
        repeat = start_repeat(labels_dir, frames_dir, args.n_repeat, args.seed, args.min_age_hours)
        clips = sorted({t["clip_id"] for t in repeat["tasks"]})
    app = App(clips, manifest, frames_dir, cache.CACHE_DIR, labels_dir, repeat)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"label tool on {url} ({len(clips)} clips). Ctrl+C to stop; every change autosaves.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
