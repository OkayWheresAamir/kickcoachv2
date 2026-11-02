"""The norm extractor's deterministic QA flags and E5's scoring."""

from __future__ import annotations

from eval import e5_norms
from kickcoach.compare.norms import NormRow, Stat
from tools.extract_norms import qa_flags

TEXT = "Table 2 ...\nKnee flexion 156.2 (10.1) 162.2 (8.8)\nSupporting foot lateral dis-\nplacement"


def test_quote_found_through_layout_noise():
    r = {"quote": "Knee  flexion 156.2 (10.1)", "mean": 156.2, "sd": 10.1, "min": None, "max": None}
    assert qa_flags(r, TEXT) == []
    assert qa_flags({"quote": "Supporting foot lateral dis- placement"}, TEXT) == []


def test_flags_missing_quote_and_numbers():
    r = {"quote": "Knee flexion 150.0 (10.1)", "mean": 150.0, "sd": 10.1}
    flags = qa_flags(r, TEXT)
    assert "quote_not_found_verbatim" in flags
    r = {"quote": "Knee flexion 156.2 (10.1)", "mean": 165.2, "sd": 10.1}
    assert qa_flags(r, TEXT) == ["mean_not_in_quote"]


def nr(norm_id, source_id="paper", **kw):
    base = dict(norm_id=norm_id, source_id=source_id, variable="kicking_knee_angle_at_impact",
                kick_type="instep", unit="deg", method="m", source="s", page_or_table="p",
                quote="q", stat=Stat(mean=1.0))  # fmt: skip
    return NormRow(**(base | kw))


def test_e5_scores_fields_hallucinations_and_recall():
    cands = [nr("N-1"), nr("N-2"), nr("N-3"), nr("N-4", source_id="ctrl")]
    verified = [
        nr("N-1", review="correct", verified=True),
        nr("N-2", review="corrected", verified=True, unit="degrees", n=7),
        nr("N-3", review="rejected"),
        nr("N-4", source_id="ctrl", review="correct", verified=True),
    ]
    s = e5_norms.score(cands, verified, missed=[{}], controls={"ctrl"})
    assert s["hallucinated"] == 2  # rejected + any control row
    assert s["supported"] == 2 and s["rows_all_fields_right"] == 1
    n = len(e5_norms.SCORED_FIELDS)
    assert (s["field_hits"], s["field_total"]) == (2 * n - 2, 2 * n)  # unit and n differ
    assert s["missed"] == 1
