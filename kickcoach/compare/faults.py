"""Fault rules from knowledge/faults.yaml, evaluated deterministically (handoff S8).

A rule fires only on a measured value. A rule whose threshold comes from a norm row is
inactive until that row is verified (D12): it is reported as inactive with the reason, so
nothing silently disappears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kickcoach.compare.engine import pipeline_stat
from kickcoach.compare.norms import NormRow

FAULTS_YAML = Path(__file__).resolve().parents[2] / "knowledge" / "faults.yaml"


@dataclass(frozen=True)
class FaultRule:
    id: str
    name: str
    feature: str
    fires_when: str  # "below" | "above" | "equals"
    threshold_source: str
    text_source: str
    template: dict
    unit: str | None = None
    threshold: float | None = None
    value: str | None = None
    norm: str | None = None
    sd: float | None = None
    citations: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)


def load_rules(path: Path = FAULTS_YAML) -> list[FaultRule]:
    rules = []
    for d in yaml.safe_load(path.read_text()):
        known = {k: d.pop(k) for k in list(d) if k in FaultRule.__dataclass_fields__}
        known["citations"] = tuple(known.get("citations") or ())
        rules.append(FaultRule(**known, extra=d))
    return rules


def resolve_threshold(rule: FaultRule, verified: list[NormRow]) -> tuple[float | None, str | None]:
    """(threshold, None), or (None, why the rule is inactive)."""
    if rule.norm is None:
        return rule.threshold, None
    row = next((r for r in verified if r.norm_id == rule.norm), None)
    if row is None:
        return None, f"its norm row {rule.norm} is not verified"
    stat, why = pipeline_stat(row, rule.unit or "")
    if stat is None:
        return None, f"norm {rule.norm} is not comparable: {why}"
    if stat.mean is None or stat.sd is None or rule.sd is None:
        return None, f"norm {rule.norm} has no mean and SD"
    k = rule.sd if rule.fires_when == "above" else -rule.sd
    return round(stat.mean + k * stat.sd, 2), None


def evaluate(
    features: list[dict], verified: list[NormRow], rules: list[FaultRule] | None = None
) -> tuple[list[dict], list[dict]]:
    """Returns (fired faults, inactive rules). Fired faults follow the MVP response shape
    (fault_id, evidence, threshold_source) plus the threshold and the rule's text source."""
    rules = load_rules() if rules is None else rules
    by_id = {f["id"]: f for f in features}
    fired, inactive = [], []
    for rule in rules:
        threshold, why = resolve_threshold(rule, verified)
        if why:
            inactive.append({"fault_id": rule.id, "reason": why})
            continue
        f = by_id.get(rule.feature)
        if f is None or f.get("status") != "measured":
            continue
        v = f["value"]
        hit = (
            v == rule.value
            if rule.fires_when == "equals"
            else (v < threshold if rule.fires_when == "below" else v > threshold)
        )
        if hit:
            fired.append(
                {
                    "fault_id": rule.id,
                    "name": rule.name,
                    "evidence": {rule.feature: v},
                    "unit": rule.unit,
                    "fires_when": rule.fires_when,
                    "threshold": rule.value if rule.fires_when == "equals" else threshold,
                    "threshold_source": rule.threshold_source,
                    "text_source": rule.text_source,
                    "citations": list(rule.citations),
                }
            )
    return fired, inactive
