"""Local page for Aamir's E7 ratings (Phase 8, DEC-040): the 20-item sample from
results/e7/items.json, one at a time, 1-5 on correctness, specificity and actionability.

    uv run python -m eval.e7_feedback prepare     # once, to build the items
    uv run python tools/rate_feedback.py           # http://127.0.0.1:8766/

The variant (LLM with passages, LLM without, template) is never sent to the page. Every
change autosaves to results/e7/aamir_ratings.json with the time it was saved; Claude never
writes a rating (CLAUDE.md rule 1).
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results" / "e7" / "items.json"
RATINGS = ROOT / "results" / "e7" / "aamir_ratings.json"
PAGE = Path(__file__).resolve().parent / "rate_ui.html"
CRITERIA = ("correctness", "specificity", "actionability")
SHOWN = ("facts", "feedback", "cited")  # what the page may see of an item; never "variant"


def payload(items_path: Path, ratings_path: Path) -> dict:
    data = json.loads(items_path.read_text())
    by_id = {i["item_id"]: i for i in data["items"]}
    ratings = json.loads(ratings_path.read_text()) if ratings_path.exists() else {}
    return {
        "rubric": data["rubric"],
        "items": [
            {
                "key": f"R{n:02d}",
                **{k: by_id[iid][k] for k in SHOWN},
                "rating": ratings.get(iid, {}),
            }
            for n, iid in enumerate(data["sample"], 1)
        ],  # fmt: skip
    }


def save(items_path: Path, ratings_path: Path, key: str, rating: dict) -> dict:
    sample = json.loads(items_path.read_text())["sample"]
    iid = sample[int(key[1:]) - 1]
    clean = {c: int(rating[c]) for c in CRITERIA if rating.get(c) in (1, 2, 3, 4, 5)}
    if str(rating.get("note", "")).strip():
        clean["note"] = str(rating["note"]).strip()[:1000]
    ratings = json.loads(ratings_path.read_text()) if ratings_path.exists() else {}
    prev = ratings.get(iid, {})
    clean["first_opened"] = prev.get("first_opened") or rating.get("opened")
    clean["saved"] = datetime.now().astimezone().isoformat(timespec="seconds")
    ratings[iid] = clean
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_path.write_text(json.dumps(ratings, indent=1))
    return clean


def make_handler(items_path: Path, ratings_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/items":
                body = json.dumps(payload(items_path, ratings_path), ensure_ascii=False)
                self._send(200, body.encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if not self.path.startswith("/api/rate/R"):
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0))
            if n > 100_000:
                self._send(413, b"too large", "text/plain")
                return
            try:
                out = save(items_path, ratings_path, self.path.rsplit("/", 1)[1],
                           json.loads(self.rfile.read(n)))  # fmt: skip
            except (ValueError, IndexError, KeyError) as e:
                self._send(400, str(e).encode(), "text/plain")
                return
            self._send(200, json.dumps(out).encode(), "application/json")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not ITEMS.exists():
        raise SystemExit(
            "no results/e7/items.json: run `uv run python -m eval.e7_feedback prepare`"
        )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(ITEMS, RATINGS))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"rating page on {url}. Ctrl+C to stop; every change autosaves.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
