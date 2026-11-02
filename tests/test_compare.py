"""Norm rows, the comparison engine and the fault rules, on synthetic inputs."""

from __future__ import annotations

import re

import pytest

from kickcoach.compare import engine, faults, norms
from kickcoach.compare.norms import NormRow, Population, Stat


def row(norm_id="N-900", **kw) -> NormRow:
    base = dict(
        norm_id=norm_id, source_id="test", variable="kicking_knee_angle_at_impact",
        kick_type="instep", unit="deg", method="3D video (DLT)",
        angle_convention="interior_180_is_straight", source="Test (2000)", page_or_table="p. 1",
        quote="q", stat=Stat(mean=150.0, sd=10.0), verified=True, review="correct",
    )  # fmt: skip
    return NormRow(**(base | kw))


def feat(fid, value, unit="deg", status="measured"):
    return {"id": fid, "value": value, "unit": unit, "status": status}


# -- norms.csv ------------------------------------------------------------------------------
def test_csv_round_trip(tmp_path):
    r = row(population=Population(level="elite", age_group="adult", sex="male"), n=7,
            condition="30 deg approach", notes="a, b")  # fmt: skip
    p = tmp_path / "n.csv"
    norms.save([r], p)
    assert norms.load(p) == [r]


def test_only_verified_correct_rows_are_used(tmp_path):
    p = tmp_path / "n.csv"
    norms.save(
        [
            row("N-1"),
            row("N-2", verified=False, review="pending"),
            row("N-3", verified=False, review="rejected"),
            row("N-4", review="corrected"),
        ],
        p,
    )
    assert [r.norm_id for r in norms.verified_norms(p)] == ["N-1", "N-4"]


def test_committed_norms_parse_and_are_all_reviewed():
    rows = norms.load()
    assert rows, "knowledge/norms.csv is missing"
    assert all(r.review != "pending" for r in rows)
    assert {r.variable for r in rows} <= set(norms.VARIABLES)


# -- conventions and units ------------------------------------------------------------------
def test_flexion_rows_convert_to_interior_angle():
    r = row(variable="kicking_knee_max_flexion", angle_convention="flexion_0_is_straight",
            stat=Stat(mean=102.71, sd=6.16, min=90.0, max=110.0))  # fmt: skip
    s, why = engine.pipeline_stat(r, "deg")
    assert why is None
    assert s.mean == pytest.approx(77.29) and s.sd == 6.16
    assert (s.min, s.max) == (70.0, 90.0)  # the ends swap


def test_unclear_convention_and_unit_mismatch_are_not_comparable():
    assert engine.pipeline_stat(row(angle_convention="unclear"), "deg")[0] is None
    cm = row(variable="support_foot_posterior_distance", unit="cm",
             angle_convention="not_applicable")  # fmt: skip
    s, why = engine.pipeline_stat(cm, "leg_lengths")
    assert s is None and "cm" in why


# -- comparisons ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, verdict", [(139.9, "below_norm"), (140.0, "within_norm"), (160.1, "above_norm")]
)
def test_verdict_uses_mean_plus_minus_one_sd(value, verdict):
    (c,) = engine.compare([feat("knee_angle_contact", value)], [row()])
    assert c["verdict"] == verdict
    assert c["norm_band"] == [140.0, 160.0]
    assert c["band_rule"].startswith("heuristic")


def test_range_rows_use_min_max_and_velocity_gets_the_2d_caveat():
    r = row(variable="peak_knee_extension_velocity", unit="deg·s-1", method="unclear",
            angle_convention="not_applicable", stat=Stat(min=860.0, max=1720.0))  # fmt: skip
    (c,) = engine.compare([feat("peak_knee_ext_velocity", 1313.5, "deg/s")], [r])
    assert c["verdict"] == "within_norm"
    assert any("Nunome & Ikegami" in x for x in c["caveats"])


def test_no_norm_and_not_measured():
    out = engine.compare(
        [
            feat("trunk_lean_contact", -12.0),
            feat("knee_angle_contact", None, status="not_measured"),
        ],
        [row()],
    )
    assert out == [
        {"feature_id": "trunk_lean_contact", "norm_ref": None, "verdict": "no_comparable_norm",
         "reason": "no verified norm row for this feature"}
    ]  # fmt: skip


def test_ranking_prefers_the_players_level_then_adults():
    rows = [
        row("N-1", population=Population(level="recreational")),
        row("N-2", population=Population(level="elite"), n=40),
        row("N-3", population=Population(level="elite", age_group="youth (13 yr)"), n=90),
    ]
    assert engine.compare([feat("knee_angle_contact", 150)], rows, "advanced")[0]["norm_ref"] == (
        "N-2"
    )
    c = engine.compare([feat("knee_angle_contact", 150)], rows, "beginner")[0]
    assert c["norm_ref"] == "N-1" and c["also_matching"] == ["N-2", "N-3"]


# -- fault rules ----------------------------------------------------------------------------
def rule(**kw) -> faults.FaultRule:
    base = dict(id="F-T", name="t", feature="trunk_lean_contact", fires_when="below",
                threshold=-10.0, unit="deg", threshold_source="heuristic: test",
                text_source="heuristic", template={})  # fmt: skip
    return faults.FaultRule(**(base | kw))


@pytest.mark.parametrize("value, fires", [(-10.1, True), (-10.0, False), (5.0, False)])
def test_threshold_rule(value, fires):
    fired, _ = faults.evaluate([feat("trunk_lean_contact", value)], [], [rule()])
    assert bool(fired) is fires
    if fires:
        assert fired[0]["evidence"] == {"trunk_lean_contact": value}


def test_rule_never_fires_on_not_measured():
    fired, _ = faults.evaluate(
        [feat("trunk_lean_contact", None, status="not_measured")], [], [rule()]
    )
    assert fired == []


def test_categorical_rule():
    r = rule(feature="segment_sequence_order", fires_when="equals", value="distal_to_proximal",
             threshold=None, unit=None)  # fmt: skip
    f = feat("segment_sequence_order", "distal_to_proximal", "order")
    assert faults.evaluate([f], [], [r])[0][0]["threshold"] == "distal_to_proximal"
    f["value"] = "simultaneous"
    assert faults.evaluate([f], [], [r])[0] == []


def test_norm_based_threshold_and_inactive_without_verification():
    r = rule(feature="max_knee_flexion_backswing", fires_when="above", threshold=None,
             norm="N-032", sd=2)  # fmt: skip
    n = row("N-032", variable="kicking_knee_max_flexion", angle_convention="flexion_0_is_straight",
            stat=Stat(mean=102.71, sd=6.16))  # fmt: skip
    assert faults.resolve_threshold(r, [n]) == (89.61, None)
    fired, _ = faults.evaluate([feat("max_knee_flexion_backswing", 90.0)], [n], [r])
    assert fired and fired[0]["threshold"] == 89.61
    fired, inactive = faults.evaluate([feat("max_knee_flexion_backswing", 99.0)], [], [r])
    assert fired == [] and "not verified" in inactive[0]["reason"]


def test_every_committed_rule_is_cited_or_heuristic():
    rules = faults.load_rules()
    assert 3 <= len(rules) <= 5
    for r in rules:
        src = r.threshold_source
        assert src.startswith("heuristic:") or re.search(r"\b(19|20)\d{2}\b", src), r.id
        assert r.text_source.startswith("heuristic") or re.search(
            r"\b(19|20)\d{2}\b", r.text_source
        )
        assert {"what", "why_it_matters", "cue", "drill"} <= set(r.template)
        # Template text holds no numbers except {value} and citation years (CLAUDE.md rule 2).
        for text in r.template.values():
            bare = re.sub(r"\b(19|20)\d{2}\)", ")", text.replace("{value}", ""))
            assert not re.search(r"\d", bare), (r.id, text)


def test_committed_rules_are_active_with_the_committed_norms():
    _, inactive = faults.evaluate([], norms.verified_norms())
    assert inactive == []
