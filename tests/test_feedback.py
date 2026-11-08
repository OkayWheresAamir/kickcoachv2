"""The feedback validator (D13), the retry and the template fallback, with fake LLM clients."""

from __future__ import annotations

import json

import pytest

from kickcoach.compare import faults
from kickcoach.feedback import generate as gen
from kickcoach.feedback.validate import numbers_in, validate
from kickcoach.llm import NoApiKey, RateLimited

FEATURES = [
    {"id": "trunk_lean_contact", "value": -11.6, "unit": "deg", "status": "measured"},
    {"id": "max_knee_flexion_backswing", "value": 98.04, "unit": "deg", "status": "measured"},
    {"id": "peak_knee_ext_velocity", "value": None, "unit": "deg/s", "status": "not_measured",
     "reason": "frame timing not trusted"},
]  # fmt: skip
FIRED = [
    {"fault_id": "F-LEAN-BACK", "name": "Leaning back", "evidence": {"trunk_lean_contact": -11.6},
     "unit": "deg", "fires_when": "below", "threshold": -10.0,
     "threshold_source": "heuristic: test", "text_source": "heuristic", "citations": []},
]  # fmt: skip


def item(**kw):
    base = {"fault_id": "F-LEAN-BACK", "what": "You leaned back 12 deg at contact.",
            "why_it_matters": "The ball rises.", "cue": "Chest over the ball.",
            "drill": "Wall strikes, freezing at contact.", "citations": []}  # fmt: skip
    return base | kw


def ctx(**kw):
    c, _ = gen.build_context(FEATURES, FIRED, [], [])
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# -- validator ------------------------------------------------------------------------------
def test_valid_answer_passes_and_rounding_is_allowed():
    fb, errors = validate({"items": [item()]}, ctx())
    assert errors == [] and fb.items[0].fault_id == "F-LEAN-BACK"


@pytest.mark.parametrize(
    "raw, fragment",
    [
        (None, "not valid JSON"),
        ({"items": [{"fault_id": "F-LEAN-BACK"}]}, "schema"),
        ({"items": [item(extra="x")]}, "schema"),
        ({"items": []}, "exactly one item"),
        ({"items": [item(), item()]}, "exactly one item"),
        ({"items": [item(), item(fault_id="F-OTHER")]}, "not detected"),
        ({"items": [item(citations=["made-up-2024"])]}, "not one of the given IDs"),
        ({"items": [item(what="You leaned back 25 deg.")]}, "number 25"),
        ({"items": [item(drill="Do 3 sets of 10 strikes.")]}, "number 3"),
        ({"items": [item(why_it_matters="It lowers your knee velocity.")]}, "was not measured"),
    ],
)
def test_validator_rejects(raw, fragment):
    _, errors = validate(raw, ctx())
    assert any(fragment in e for e in errors), errors


def test_numbers_in_text():
    assert numbers_in("−12 deg, 0.32 leg lengths, 1,092.23 deg/s, N-032, item 3") == [
        -12.0, 0.32, 1092.23, 3.0,
    ]  # fmt: skip


def test_citation_years_and_ids_are_not_invented_numbers():
    c = ctx(citation_ids={"kellis_katis_2007"})
    c.allow(2007, "year")
    fb, errors = validate(
        {"items": [item(why_it_matters="Kicking studies agree (Kellis & Katis 2007; N-032).",
                        citations=["kellis_katis_2007"])]},
        c,
    )  # fmt: skip
    assert errors == [], errors


def test_committed_templates_pass_the_validator():
    rules = faults.load_rules()
    feats = [
        {"id": r.feature, "value": "distal_to_proximal" if r.fires_when == "equals" else -0.5,
         "unit": r.unit, "status": "measured"}
        for r in rules
    ]  # fmt: skip
    fired, _ = faults.evaluate(feats, [], [r for r in rules if r.norm is None])
    tmpl = gen.template_feedback(fired, rules)
    c, _ = gen.build_context(feats, fired, [], [])
    assert validate(tmpl.model_dump(), c)[1] == []


# -- generation: retry and fallback ---------------------------------------------------------
class FakeClient:
    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def generate_json(self, *, prompt, **kw):
        self.prompts.append(prompt)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return {"parsed": a, "text": json.dumps(a), "key": f"k{len(self.prompts)}", "cached": False}


def run(client, **kw):
    return gen.generate_feedback(FEATURES, FIRED, [], [], client=client, **kw)


def test_llm_answer_used_when_valid():
    out = run(FakeClient([{"items": [item()]}]))
    assert out["generated_by"] == "llm" and out["validator_passed"] is True
    assert len(out["attempts"]) == 1


def test_one_retry_with_the_errors_then_success():
    client = FakeClient([{"items": [item(what="You leaned back 25 deg.")]}, {"items": [item()]}])
    out = run(client)
    assert out["generated_by"] == "llm" and len(out["attempts"]) == 2
    assert "number 25" in client.prompts[1] and "rejected" in client.prompts[1]


def test_two_failures_fall_back_to_the_template():
    bad = {"items": [item(citations=["nope"])]}
    out = run(FakeClient([bad, bad]))
    assert out["generated_by"] == "template" and out["validator_passed"] is True
    assert len(out["attempts"]) == 2 and "both LLM attempts" in out["fallback_reason"]
    assert out["items"][0]["what"].startswith("Your trunk was leaning back (12 deg")


@pytest.mark.parametrize("exc", [NoApiKey("no key"), RateLimited("429")])
def test_no_key_or_quota_gives_template(exc):
    out = run(FakeClient([exc]))
    assert out["generated_by"] == "template" and out["items"]
    assert out["attempts"] == []


def test_no_faults_means_no_llm_call():
    client = FakeClient([])
    out = gen.generate_feedback(FEATURES, [], [], [], client=client)
    assert out["items"] == [] and client.prompts == []


def test_runs_without_an_api_key(tmp_path, monkeypatch):
    """CLAUDE.md rule 5 with the real client: no key in the environment or in .env."""
    import kickcoach.llm.gemini as gemini

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "ROOT", tmp_path)  # no .env there
    client = gemini.GeminiClient(gemini.DEV_MODEL, tmp_path / "cache")
    out = run(client)
    assert out["generated_by"] == "template" and "no GEMINI_API_KEY" in out["fallback_reason"]


# -- checks 6 and 7, from E7's ratings (Phase 9, DEC-042) -------------------------------------
def ctx_with(**values):
    feats = [dict(f) for f in FEATURES]
    for f in feats:
        f["value"] = values.get(f["id"], f["value"])
    feats.append({"id": "plant_foot_offset", "value": values.get("plant_foot_offset", -0.32),
                  "unit": "leg_lengths", "status": "measured"})  # fmt: skip
    fired = [f | {"evidence": {"trunk_lean_contact": values.get("trunk_lean_contact", -11.6)}}
             for f in FIRED]  # fmt: skip
    return gen.build_context(feats, fired, [], [])[0]


@pytest.mark.parametrize(
    "text",
    [
        "Your knee bend in the backswing was 98 deg.",  # E7 R13, word for word
        "Your knee bent to 98 degrees in the backswing, which is too little bend.",  # c03 v3
        "Your knee flexion was 98 degrees.",
    ],
)
def test_a_knee_interior_angle_called_bend_is_rejected(text):
    _, errors = validate({"items": [item(what=text)]}, ctx())
    assert any("interior angle" in e for e in errors), errors


@pytest.mark.parametrize(
    "text",
    [
        "Your knee angle reached 98 degrees during the backswing.",  # the wording v3 used on c20
        "Your knee stayed too straight, reaching an angle of 98 degrees instead of bending more.",
        "Bend your knee more as you swing back.",
        "You leaned back 12 deg, so the bend in your knee mattered less.",  # 12 isn't a knee angle
    ],
)
def test_knee_wording_with_angle_or_without_a_knee_number_passes(text):
    _, errors = validate({"items": [item(what=text)]}, ctx())
    assert not any("interior angle" in e for e in errors), errors


def test_a_lean_the_kick_did_not_have_is_rejected():
    # E7 R11: a plant-foot item on a kick whose trunk leaned 5.36 deg forward
    why = "Landing too far back makes your body lean away from the ball and reduces your power."
    c = ctx_with(trunk_lean_contact=5.36)
    _, errors = validate({"items": [item(what="Your plant foot landed behind the ball.",
                                         why_it_matters=why)]}, c)  # fmt: skip
    assert any("'lean away'" in e and "5.36" in e for e in errors), errors


@pytest.mark.parametrize(
    "values, field, text, rejected",
    [
        ({}, "why_it_matters", "Leaning back makes the ball rise.", False),  # -11.6: agrees
        ({}, "drill", "Wall strikes keeping your chest leaning forward.", False),  # advice
        ({}, "what", "You leaned forward at contact.", True),
        ({"trunk_lean_contact": 5.36}, "what", "You leaned back at contact.", True),
        ({"plant_foot_offset": 0.14}, "what", "Your plant foot landed behind the ball.", True),
        ({}, "what", "Your plant foot landed behind the ball.", False),  # -0.32: agrees
        ({}, "what", "Your plant foot landed ahead of the ball.", True),
        ({}, "why_it_matters", "Plant beside or just ahead of the ball.", False),  # advice
    ],
)
def test_direction_claims_are_checked_against_the_measured_sign(values, field, text, rejected):
    _, errors = validate({"items": [item(**{field: text})]}, ctx_with(**values))
    assert any("this kick's measurement shows" in e for e in errors) == rejected, errors


def test_shallow_backswing_template_passes_with_its_norm_threshold():
    (rule,) = [r for r in faults.load_rules() if r.id == "F-SHALLOW-BACKSWING"]
    feats = [{"id": rule.feature, "value": 100.89, "unit": "deg", "status": "measured"}]
    fired = [{"fault_id": rule.id, "name": rule.name, "evidence": {rule.feature: 100.89},
              "unit": "deg", "fires_when": "above", "threshold": 89.61,
              "threshold_source": rule.threshold_source, "text_source": rule.text_source,
              "citations": list(rule.citations)}]  # fmt: skip
    tmpl = gen.template_feedback(fired, [rule])
    c, _ = gen.build_context(feats, fired, [], [])
    assert validate(tmpl.model_dump(), c)[1] == []


def test_rag_prompt_asks_for_the_value_and_angle_wording():
    facts = {"player_level": None, "coaching_passages": [{"id": "K-01", "text": "x"}]}
    p = gen.build_prompt(facts)
    assert "with the fault's measured value" in p and "never an amount of" in p
    assert "with the fault's measured value" not in gen.build_prompt({"player_level": None})
