"""SQLite storage (handoff D16): sessions, features, faults, feedback, artifacts, and the
self-history deltas a session is compared with.

stdlib sqlite3 only; one connection per call, so the API can use it from any thread. The
database file is gitignored (data/kickcoach.db).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "kickcoach.db"
# heuristic: "your last few sessions" in the MVP spec's example ("vs last 3 sessions").
HISTORY_WINDOW = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    player TEXT NOT NULL,
    level TEXT,
    clip_name TEXT,
    accepted INTEGER NOT NULL,
    quality TEXT NOT NULL,
    events TEXT NOT NULL,
    pose_model TEXT,
    inputs TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS features (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    feature_id TEXT NOT NULL,
    value REAL,
    value_text TEXT,
    unit TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS faults (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    fault_id TEXT NOT NULL,
    evidence TEXT NOT NULL,
    threshold_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    generated_by TEXT NOT NULL,
    validator_passed INTEGER,
    items TEXT NOT NULL,
    attempts TEXT NOT NULL,
    fallback_reason TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS features_by_session ON features(session_id);
CREATE INDEX IF NOT EXISTS sessions_by_player ON sessions(player, id);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def save_session(
    con: sqlite3.Connection,
    *,
    player: str,
    level: str | None,
    clip_name: str | None,
    result: dict,
    inputs: dict,
    artifacts: dict[str, str] | None = None,
) -> int:
    """Store one analysed clip. ``result`` is the pipeline output plus ``faults`` and
    ``feedback``; returns the session id."""
    with con:
        cur = con.execute(
            "INSERT INTO sessions (created_at, player, level, clip_name, accepted, quality, "
            "events, pose_model, inputs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                player,
                level,
                clip_name,
                int(bool(result["quality"]["accepted"])),
                json.dumps(result["quality"]),
                json.dumps(result["events"]),
                result.get("pose_model"),
                json.dumps(inputs),
            ),
        )
        sid = cur.lastrowid
        for f in result.get("features", []):
            v = f.get("value")
            con.execute(
                "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid, f["id"],
                    float(v) if isinstance(v, float | int) else None,
                    v if isinstance(v, str) else None,
                    f.get("unit"), f["status"], f.get("reason"),
                    json.dumps({k: f[k] for k in ("frame", "keypoint_score", "extra") if k in f}),
                ),
            )  # fmt: skip
        for fault in result.get("faults", []):
            con.execute(
                "INSERT INTO faults VALUES (?, ?, ?, ?)",
                (sid, fault["fault_id"], json.dumps(fault["evidence"]), fault["threshold_source"]),
            )
        fb = result.get("feedback")
        if fb is not None:
            con.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sid, fb["generated_by"],
                    None if fb.get("validator_passed") is None else int(fb["validator_passed"]),
                    json.dumps(fb["items"]), json.dumps(fb.get("attempts", [])),
                    fb.get("fallback_reason"),
                ),
            )  # fmt: skip
        for kind, path in (artifacts or {}).items():
            con.execute("INSERT INTO artifacts VALUES (?, ?, ?)", (sid, kind, path))
    return sid


def history_delta(
    con: sqlite3.Connection, player: str, session_id: int, window: int = HISTORY_WINDOW
) -> dict[str, dict]:
    """For each numeric feature measured in this session, the change against the mean of the
    same player's last ``window`` sessions in which it was measured. Earlier sessions only."""
    current = {
        r["feature_id"]: r["value"]
        for r in con.execute(
            "SELECT feature_id, value FROM features WHERE session_id = ? AND status = "
            "'measured' AND value IS NOT NULL",
            (session_id,),
        )
    }
    out = {}
    for fid, value in current.items():
        prev = [
            r["value"]
            for r in con.execute(
                "SELECT f.value FROM features f JOIN sessions s ON s.id = f.session_id "
                "WHERE s.player = ? AND s.id < ? AND f.feature_id = ? AND f.status = "
                "'measured' AND f.value IS NOT NULL ORDER BY s.id DESC LIMIT ?",
                (player, session_id, fid, window),
            )
        ]
        if not prev:
            continue
        mean = sum(prev) / len(prev)
        out[fid] = {
            "delta": round(value - mean, 2),
            "vs_mean_of": len(prev),
            "text": f"{value - mean:+.1f} vs last {len(prev)} session{'s' * (len(prev) > 1)}",
        }
    return out


def list_sessions(con: sqlite3.Connection, player: str | None = None, limit: int = 30) -> list:
    q = "SELECT * FROM sessions" + (" WHERE player = ?" if player else "")
    rows = con.execute(q + " ORDER BY id DESC LIMIT ?", ((player,) if player else ()) + (limit,))
    out = []
    for s in rows:
        feats = {
            r["feature_id"]: r["value"] if r["value"] is not None else r["value_text"]
            for r in con.execute(
                "SELECT feature_id, value, value_text FROM features WHERE session_id = ? AND "
                "status = 'measured'",
                (s["id"],),
            )
        }
        faults = [r["fault_id"] for r in con.execute(
            "SELECT fault_id FROM faults WHERE session_id = ?", (s["id"],))]  # fmt: skip
        out.append(
            {
                "session_id": s["id"],
                "created_at": s["created_at"],
                "player": s["player"],
                "level": s["level"],
                "clip_name": s["clip_name"],
                "accepted": bool(s["accepted"]),
                "features": feats,
                "faults": faults,
            }
        )
    return out
