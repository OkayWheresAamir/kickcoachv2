"""The coach chat's tools (DEC-008, DEC-043): read-only functions over one player's stored
sessions, the norm comparison and the coaching corpus.

The model chooses which tools to call; everything they return is computed here. Trends come
with their differences already taken and compared with the measurement's error, so the
model restates numbers and never derives them (CLAUDE.md rule 3). The player is fixed when
the chat starts, so the model can't read anyone else's sessions.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from functools import cache
from pathlib import Path

from kickcoach.compare import engine, faults, norms
from kickcoach.feedback.generate import DEFINITIONS, FEATURE_UNITS
from kickcoach.retrieval import corpus as coaching
from kickcoach.retrieval.embed import Embedder
from kickcoach.retrieval.retrieve import KICK_TYPES, filter_chunks, rank

ROOT = Path(__file__).resolve().parents[2]
E2_CSV = ROOT / "results" / "e2.csv"
FEATURE_IDS = list(DEFINITIONS)
FAULT_IDS = [r.id for r in faults.load_rules()]
# heuristic: enough sessions to see a trend without flooding the context.
DEFAULT_LAST_N, MAX_LAST_N = 5, 20
PASSAGES = 3  # E6 scores retrieval at 3


DECLARATIONS = [
    {
        "name": "get_history",
        "description": (
            "The player's stored kick sessions, newest first: date, whether the clip was "
            "accepted (or why it was refused), each feature's value or why it wasn't measured, "
            "and the faults found. With feature_id, also that feature's trend over those "
            "sessions: values oldest first, the change from first to last, each step's change, "
            "and whether the change is larger than the measurement's typical error."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "feature_id": {"type": "string", "enum": FEATURE_IDS,
                               "description": "The feature whose trend to compute."},
                "last_n": {"type": "integer", "minimum": 1, "maximum": MAX_LAST_N,
                           "description": f"How many recent sessions (default "
                                          f"{DEFAULT_LAST_N})."},
            },
        },
    },
    {
        "name": "get_session_features",
        "description": (
            "Everything measured in one session: each feature's value, unit and definition, "
            "or why it wasn't measured; the faults found with their thresholds and where each "
            "threshold comes from; or why the clip was refused."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "integer",
                               "description": "Default: the latest session."},
            },
        },
    },
    {
        "name": "compare_to_norms",
        "description": (
            "Compares one feature of one session with published study values (verified norm "
            "rows): the verdict, the study's mean, SD and band, its population and caveats; "
            "and the fault rule on that feature, if any, with its threshold and source."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "feature_id": {"type": "string", "enum": FEATURE_IDS},
                "session_id": {"type": "integer",
                               "description": "Default: the latest session."},
            },
            "required": ["feature_id"],
        },
    },
    {
        "name": "search_drills",
        "description": (
            "Searches short coaching passages (cues, drills, explanations) quoted from "
            f"published coaching sources. Returns the best {PASSAGES} with their IDs and "
            "sources. Give fault_id when the question is about one of these faults: the "
            "search is then limited to passages about it, which works better."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in words."},
                "fault_id": {"type": "string", "enum": FAULT_IDS},
            },
            "required": ["query"],
        },
    },
]  # fmt: skip


@cache
def typical_errors() -> dict[str, dict]:
    """E2's mean absolute error of the pipeline against Aamir's labels, per feature. Read
    from results/e2.csv, which eval/e2_angles.py writes, so the number is never copied."""
    if not E2_CSV.exists():
        return {}
    errs: dict[str, list[float]] = {}
    with open(E2_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["view"] == "end_to_end" and r["system"] == "v2_pipeline" and r["err"]:
                errs.setdefault(r["feature"], []).append(abs(float(r["err"])))
    return {
        fid: {"value": round(sum(v) / len(v), 2),
              "source": f"E2: mean absolute error against hand labels, {len(v)} clips"}
        for fid, v in errs.items()
    }  # fmt: skip


def _r(x):
    return round(x, 2) if isinstance(x, float) else x


class Tools:
    def __init__(
        self,
        con: sqlite3.Connection,
        player: str,
        *,
        verified: list[norms.NormRow] | None = None,
        embedder: Embedder | None = None,
        level: str | None = None,
    ) -> None:
        self.con, self.player, self.level = con, player, level
        self.verified = norms.verified_norms() if verified is None else verified
        self._embedder = embedder

    # -- dispatch ---------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        """A tool's result, or {"error": ...} that goes back to the model as its answer."""
        fn = {d["name"]: getattr(self, d["name"]) for d in DECLARATIONS}.get(name)
        if fn is None:
            return {"error": f"no tool called {name!r}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except ValueError as e:
            return {"error": str(e)}

    # -- data access ------------------------------------------------------------------------
    def _session_ids(self) -> list[int]:
        rows = self.con.execute(
            "SELECT id FROM sessions WHERE player = ? ORDER BY id DESC", (self.player,)
        )
        return [r["id"] for r in rows]

    def _session(self, session_id: int | None) -> dict:
        ids = self._session_ids()
        if not ids:
            raise ValueError("this player has no stored sessions yet")
        sid = ids[0] if session_id is None else int(session_id)
        if sid not in ids:
            raise ValueError(f"no session {sid} for this player; their sessions are {ids}")
        s = self.con.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        feats = [
            {"id": r["feature_id"],
             "value": r["value"] if r["value"] is not None else r["value_text"],
             "unit": r["unit"], "status": r["status"], "reason": r["reason"]}
            for r in self.con.execute("SELECT * FROM features WHERE session_id = ?", (sid,))
        ]  # fmt: skip
        q = json.loads(s["quality"])
        return {"session_id": sid, "date": s["created_at"][:10], "accepted": bool(s["accepted"]),
                "quality": q, "features": feats}  # fmt: skip

    def _faults(self, features: list[dict]) -> list[dict]:
        # Re-evaluated with today's rules, like the norm comparison, rather than read back.
        fired, _ = faults.evaluate(features, self.verified)
        return [{k: _r(f[k]) if k != "evidence" else {e: _r(v) for e, v in f[k].items()}
                 for k in ("fault_id", "name", "evidence", "unit", "fires_when", "threshold",
                           "threshold_source")} for f in fired]  # fmt: skip

    @staticmethod
    def _refused(s: dict) -> dict:
        return {
            "refused_because": s["quality"].get("reasons", []),
            "note": "the clip was refused by the quality check, so nothing was measured",
        }

    # -- tools ------------------------------------------------------------------------------
    def get_session_features(self, session_id: int | None = None) -> dict:
        s = self._session(session_id)
        out = {"session_id": s["session_id"], "date": s["date"], "accepted": s["accepted"]}
        if not s["accepted"]:
            return out | self._refused(s)
        out["features"] = [
            {"id": f["id"], "status": f["status"], "definition": DEFINITIONS.get(f["id"]),
             **({"value": _r(f["value"]), "unit": f["unit"]} if f["status"] == "measured"
                else {"reason": f["reason"]})}
            for f in s["features"]
        ]  # fmt: skip
        out["faults"] = self._faults(s["features"])
        if s["quality"].get("warnings"):
            out["warnings"] = s["quality"]["warnings"]
        return out

    def get_history(self, feature_id: str | None = None, last_n: int | None = None) -> dict:
        if feature_id is not None and feature_id not in FEATURE_IDS:
            raise ValueError(f"unknown feature {feature_id!r}; one of {FEATURE_IDS}")
        n = min(max(int(last_n or DEFAULT_LAST_N), 1), MAX_LAST_N)
        ids = self._session_ids()
        sessions = [self._session(sid) for sid in ids[:n]]
        rows = []
        for s in sessions:
            row = {"session_id": s["session_id"], "date": s["date"], "accepted": s["accepted"]}
            if not s["accepted"]:
                row |= self._refused(s)
            else:
                # The reason is included because, without it, the model wrote its own ("not
                # measured (insufficient markers on shank)") in E9 (DEC-043).
                row["values"] = {f["id"]: _r(f["value"]) if f["status"] == "measured"
                                 else f"not measured: {f['reason'] or 'no reason recorded'}"
                                 for f in s["features"]}  # fmt: skip
                row["faults"] = [f["fault_id"] for f in self._faults(s["features"])]
            rows.append(row)
        # Run 1 of E9 returned bare numbers here, and the model wrote a leg-length offset in
        # metres ("-0.32 m"); the units are now part of the result.
        out = {"player": self.player, "sessions_stored": len(ids), "units": FEATURE_UNITS,
               "sessions": rows}  # fmt: skip
        if feature_id is not None:
            out["trend"] = self._trend(feature_id, list(reversed(sessions)))
        return out

    def _trend(self, fid: str, oldest_first: list[dict]) -> dict:
        points, values = [], []
        unit = None
        for s in oldest_first:
            f = next((f for f in s["features"] if f["id"] == fid), None)
            p = {"session_id": s["session_id"], "date": s["date"]}
            if not s["accepted"]:
                p["not_measured"] = "clip refused"
            elif f is None or f["status"] != "measured":
                p["not_measured"] = (f or {}).get("reason") or "not measured"
            else:
                p["value"], unit = _r(f["value"]), f["unit"]
                if isinstance(f["value"], float | int):
                    values.append(float(f["value"]))
            points.append(p)
        out = {
            "feature_id": fid,
            "definition": DEFINITIONS.get(fid),
            "unit": unit,
            "points_oldest_first": points,
            "measured_in": sum("value" in p for p in points),
        }
        if len(values) >= 2:
            change = values[-1] - values[0]
            err = typical_errors().get(fid)
            out |= {
                "change_first_to_last": round(change, 2),
                "step_changes": [round(b - a, 2)
                                 for a, b in zip(values[:-1], values[1:], strict=True)],
                "typical_error": err or {"value": None,
                                         "why": "this feature's accuracy hasn't been measured"},
                # heuristic: one MAE. A smaller change can't be told apart from measurement
                # error on a single clip per session.
                "change_exceeds_typical_error": None if err is None
                else abs(change) > err["value"],
            }  # fmt: skip
            # Said in words, because the page check after E9's run 2 got "your backswing is
            # getting shallower" for a change smaller than the error (DEC-043).
            out["reading"] = (
                "this feature's accuracy hasn't been measured, so the change can't be compared "
                "with its error"
                if err is None
                else "the change is larger than the typical measurement error"
                if abs(change) > err["value"]
                else "the change is smaller than the typical measurement error, so it can't be "
                "told apart from measurement noise: don't call it a trend"
            )
        elif values:
            out["note"] = "measured in only one of these sessions, so there is no trend"
        else:
            out["note"] = "not measured in any of these sessions"
        return out

    def compare_to_norms(self, feature_id: str, session_id: int | None = None) -> dict:
        if feature_id not in FEATURE_IDS:
            raise ValueError(f"unknown feature {feature_id!r}; one of {FEATURE_IDS}")
        s = self._session(session_id)
        out = {"session_id": s["session_id"], "feature_id": feature_id,
               "definition": DEFINITIONS.get(feature_id)}  # fmt: skip
        if not s["accepted"]:
            return out | self._refused(s)
        f = next((f for f in s["features"] if f["id"] == feature_id), None)
        if f is None or f["status"] != "measured":
            return out | {"status": "not_measured",
                          "reason": (f or {}).get("reason") or "not measured"}  # fmt: skip
        comp = next((c for c in engine.compare(s["features"], self.verified, self.level)
                     if c["feature_id"] == feature_id), None)  # fmt: skip
        out |= {"value": _r(f["value"]), "unit": f["unit"]}
        if comp is None:
            out["comparison"] = {
                "verdict": "no_comparable_norm",
                "reason": "no study variable maps to this feature",
            }
        else:
            out["comparison"] = {
                k: v
                for k, v in comp.items()
                if k not in ("feature_id", "value", "unit", "also_matching")
            }
        rule = next((r for r in faults.load_rules() if r.feature == feature_id), None)
        if rule is not None:
            threshold, why = faults.resolve_threshold(rule, self.verified)
            fired = {x["fault_id"] for x in self._faults(s["features"])}
            out["fault_rule"] = {
                "fault_id": rule.id,
                "name": rule.name,
                "fires_when": rule.fires_when,
                "threshold": rule.value if rule.fires_when == "equals" else threshold,
                "threshold_source": rule.threshold_source,
                "fired": rule.id in fired,
                **({"inactive": why} if why else {}),
            }
        return out  # fmt: skip

    def search_drills(self, query: str, fault_id: str | None = None) -> dict:
        query = str(query).strip()
        if not query:
            raise ValueError("query is empty")
        if fault_id is not None and fault_id not in FAULT_IDS:
            raise ValueError(f"unknown fault {fault_id!r}; one of {FAULT_IDS}")
        chunks = coaching.load()
        pool = (filter_chunks(chunks, fault_id) if fault_id
                else [c for c in chunks if c.kick_type in KICK_TYPES])  # fmt: skip
        hits = rank(pool, query, self.embedder)[:PASSAGES]
        return {
            "method": "fault filter, then embedding similarity" if fault_id
            else "embedding similarity over all instep and general passages",
            "passages": [{"id": c.id, "text": c.text, "source": f"{c.cite}, {c.where}",
                          "url": c.url, "faults": list(c.faults)} for c, _ in hits],
            "note": "quoted passages; use one only where it fits the question",
        }  # fmt: skip

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(write_cache=ROOT / "data" / "cache" / "chat_queries.npz")
        return self._embedder
