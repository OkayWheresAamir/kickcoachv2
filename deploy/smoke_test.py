"""Upload one clip to a running KickCoach (local or the hosted demo), poll the job, and print
each stage with its time and the result's headline. Exits 1 if the job fails.

    uv run --group deploy python deploy/smoke_test.py <base_url> data/raw/clips/c07.mp4
    uv run --group deploy python deploy/smoke_test.py http://127.0.0.1:8000 kick.mp4 --history '[]'

The hosted demo's base URL is in the README.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("video", type=Path)
    ap.add_argument("--history", help="the page's earlier sessions, as JSON")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    form = {"history": args.history} if args.history else {}
    t0 = time.monotonic()
    with open(args.video, "rb") as f:
        r = requests.post(f"{base}/analyse", files={"file": (args.video.name, f)}, data=form,
                          timeout=300)  # fmt: skip
    print(f"{time.monotonic() - t0:6.1f}s POST {r.status_code}", flush=True)
    r.raise_for_status()
    job, last = r.json()["job_id"], None
    while True:
        s = requests.get(f"{base}/jobs/{job}", timeout=60).json()
        if (s["state"], s["stage"]) != last:
            last = (s["state"], s["stage"])
            print(f"{time.monotonic() - t0:6.1f}s {s['state']} {s['stage'] or ''}", flush=True)
        if s["state"] in ("done", "failed"):
            break
        time.sleep(2)
    if s["state"] == "failed":
        print(f"failed: {s['error']}")
        sys.exit(1)
    d = s["result"]
    q, ev = d["quality"], d["events"]
    print(f"total {time.monotonic() - t0:.1f}s; accepted {q['accepted']} {q['reasons'] or ''}")
    print(f"contact frame {ev['contact_frame']} ({ev['contact_method']}), leg {ev['kicking_leg']}")
    for f in d["features"]:
        print(f"  {f['id']}: {f.get('value', f.get('reason'))}")
    print(f"faults {[f['fault_id'] for f in d['faults']]}; "
          f"feedback {d['feedback']['generated_by']}")  # fmt: skip
    print(f"history_delta {d['history_delta']}")


if __name__ == "__main__":
    main()
