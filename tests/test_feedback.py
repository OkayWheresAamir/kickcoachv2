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
