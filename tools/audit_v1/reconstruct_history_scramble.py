"""Reconstruct which feature each value in v1's kick_history.csv really was.

v1 sorted the per-feature deviations by severity (feedback_engine._compute_deviations),
built each history row in that order, and wrote it with a csv.DictWriter whose header
was only written for the first row (app._append_history). So every row's values appear
in *that session's* severity order under the *first session's* header.

For each row, this script tries every assignment of the 5 values to the 5 features and
keeps the ones that are self-consistent: re-running v1's severity sort on that assignment
must reproduce the order the values were written in. It then drops physically impossible
assignments (e.g. a negative speed).

Run: uv run python tools/audit_v1/reconstruct_history_scramble.py
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Copied from kickcoach-2 @ deb3c4e: reference_data.PRO_PROFILES["youth_senior"]["reference"]
# (every history row is age 19, which select_profile maps to youth_senior) and
# feedback_engine.SEVERITY_THRESHOLDS (moderate, significant). Dict order matters: it is the
# order v1 iterated in before its stable sort.
REFERENCE = {
    "knee_angle": 130.00,
    "trunk": 8.50,
    "hip_rotation": -2.50,
    "ankle_speed_pps": 1580.00,
    "knee_ang_vel_dps": 1150.00,
}
THRESHOLDS = {
    "knee_angle": (0.05, 0.12),
    "trunk": (0.15, 0.35),
    "hip_rotation": (0.20, 0.50),
    "ankle_speed_pps": (0.10, 0.25),
    "knee_ang_vel_dps": (0.10, 0.25),
}
SEVERITY_RANK = {"significant": 0, "moderate": 1, "good": 2}


def v1_order(snapshot: dict[str, float]) -> list[str]:
    devs = []
    for feature, ref in REFERENCE.items():
        pct = abs((snapshot[feature] - ref) / ref)
        moderate, significant = THRESHOLDS[feature]
        sev = "good" if pct <= moderate else "moderate" if pct <= significant else "significant"
        devs.append((feature, sev))
    devs.sort(key=lambda d: SEVERITY_RANK[d[1]])
    return [d[0] for d in devs]


def plausible(s: dict[str, float]) -> bool:
    return (
        0 <= s["knee_angle"] <= 180
        and abs(s["trunk"]) <= 90
        and abs(s["hip_rotation"]) <= 180
        and s["ankle_speed_pps"] >= 0
    )


def main() -> None:
    with open(DATA / "kick_history.csv", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0][8:]
    print(f"header feature order (from the first session): {header}\n")
    for row in rows[1:]:
        values = [float(x) for x in row[8:]]
        consistent = []
        for perm in itertools.permutations(REFERENCE):
            snap = dict(zip(perm, values, strict=True))
            if v1_order(snap) == list(perm):
                consistent.append(snap)
        good = [s for s in consistent if plausible(s)]
        print(f"frame {row[1]}: {len(consistent)} self-consistent, {len(good)} plausible")
        for s in good:
            mislabels = {
                h: f for h, v in zip(header, values, strict=True) for f in s if s[f] == v and f != h
            }
            print(f"   true values: { {k: round(v, 1) for k, v in s.items()} }")
            print(f"   written under the wrong header (header -> true feature): {mislabels}")


if __name__ == "__main__":
    main()
