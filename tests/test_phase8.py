"""Phase 8: retrieved passages in the feedback (prompt v3), E7's scoring and the rating tool."""

from __future__ import annotations

import json

import pytest

from eval import e7_feedback as e7
from kickcoach.feedback import generate as gen
from kickcoach.llm import NoApiKey
from kickcoach.retrieval import corpus
from tests.test_feedback import FEATURES, FIRED, FakeClient, item
from tools import rate_feedback as rate

CHUNKS = {c.id: c for c in corpus.load()}
PASSAGES = {"F-LEAN-BACK": [CHUNKS["K-12"], CHUNKS["K-26"]]}


def run_rag(client, monkeypatch, passages=PASSAGES):
    monkeypatch.setattr(gen, "retrieve_passages", lambda fired, rules, embedder=None: passages)
    return gen.generate_feedback(FEATURES, FIRED, [], [], client=client, retrieval=True)


def test_passages_reach_the_prompt_and_become_citable(monkeypatch):
    client = FakeClient([{"items": [item(citations=["K-12"])]}])
    out = run_rag(client, monkeypatch)
    assert out["generated_by"] == "llm" and out["validator_passed"] is True
    prompt = client.prompts[0]
    assert '"coaching_passages"' in prompt and "Copy no numbers or" in prompt
    assert CHUNKS["K-12"].text in prompt
    assert out["passages"] == {"F-LEAN-BACK": ["K-12", "K-26"]}
    assert out["sources"]["K-12"]["url"] == CHUNKS["K-12"].url


def test_a_passage_that_was_not_retrieved_cannot_be_cited(monkeypatch):
    bad = {"items": [item(citations=["K-35"])]}
    out = run_rag(FakeClient([bad, bad]), monkeypatch)
    assert out["generated_by"] == "template"
    assert "citation 'K-35' is not one of the given IDs" in out["attempts"][0]["errors"][0]


def test_without_passages_the_prompt_is_v2_word_for_word():
    facts = {"player_level": None, "faults": []}
    p = gen.build_prompt(facts)
    assert "coaching_passages" not in p
    assert "6. Short sentences a teenager would understand." in p


def test_retrieval_without_a_cached_query_and_no_key_gives_no_passages():
    class NoKey:
        def vectors(self, inputs):
            raise NoApiKey("no key")

    rules = gen.load_rules()
    assert gen.retrieve_passages(FIRED, rules, NoKey()) == {}


def test_templates_never_get_passages():
    out = gen.generate_feedback(FEATURES, FIRED, [], [], use_llm=False, retrieval=True)
    assert out["generated_by"] == "template" and "passages" not in out


@pytest.mark.parametrize(
    "a,b,expected",
    [([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0), ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0)],
)
def test_weighted_kappa_extremes(a, b, expected):
    assert e7.weighted_kappa(a, b) == pytest.approx(expected)


def test_weighted_kappa_known_value():
    # Hand-computed: o = diag(.5,.5) on {4,5} -> 1; one disagreement by 1 step out of 4.
    a, b = [4, 4, 5, 5], [4, 5, 5, 5]
    k = e7.weighted_kappa(a, b)
    assert 0 < k < 1 and k == pytest.approx(e7.weighted_kappa(b, a))


def test_judge_prompt_shows_what_aamir_sees_and_never_the_variant():
    data = json.loads(e7.ITEMS.read_text())
    it = data["items"][0]
    p = e7.judge_prompt(it)
    assert it["feedback"]["what"] in p and "variant" not in p and it["item_id"] not in p


def test_rating_payload_hides_the_variant(tmp_path):
    p = rate.payload(e7.ITEMS, tmp_path / "r.json")
    blob = json.dumps(p)
    assert len(p["items"]) == 20
    assert "llm_rag" not in blob and 'template"' not in blob and "item_id" not in blob


def test_rating_save_keeps_only_valid_scores_and_the_first_open_time(tmp_path):
    path = tmp_path / "r.json"
    rate.save(e7.ITEMS, path, "R01", {"correctness": 4, "specificity": 9, "opened": "t0"})
    rate.save(e7.ITEMS, path, "R01", {"correctness": 5, "actionability": 3, "opened": "t1",
                                     "note": "  ok "})  # fmt: skip
    (saved,) = json.loads(path.read_text()).values()
    assert saved["correctness"] == 5 and saved["actionability"] == 3
    assert "specificity" not in saved and saved["note"] == "ok" and saved["first_opened"] == "t0"


def test_e7_scores_before_any_rating_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(e7, "RATINGS", tmp_path / "none.json")
    rows, md = e7.score()
    assert len(rows) == 54 and "Not computed yet" in md
