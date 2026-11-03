"""The coaching corpus, its embeddings cache and retrieval (Phase 8, E6)."""

from __future__ import annotations

import itertools
from statistics import mean

import numpy as np
import pytest
import yaml

from eval import e6_retrieval
from kickcoach.compare.faults import load_rules
from kickcoach.retrieval import corpus
from kickcoach.retrieval.embed import Embedder, doc_input, query_input
from kickcoach.retrieval.retrieve import (
    NEEDS,
    expected_random_metrics,
    filter_chunks,
    query_text,
    rank,
)
from tools import coaching_sources as cs

CHUNKS = corpus.load()


def test_corpus_is_well_formed():
    ids = [c.id for c in CHUNKS]
    assert len(ids) == len(set(ids)) and 30 <= len(ids) <= 60  # handoff D14: 30-60 chunks
    rule_ids = {r.id for r in load_rules()}
    for c in CHUNKS:
        assert c.kick_type in corpus.KICK_TYPES and c.url.startswith("https://")
        assert set(c.faults) <= rule_ids, c.id
        assert c.source in cs.BY_ID, c.id


def test_copyrighted_excerpts_stay_short():
    for c in CHUNKS:
        if cs.BY_ID[c.source]["group"] == "C":
            assert len(c.text.split()) <= 60, c.id


def test_every_chunk_is_verbatim_where_the_source_is_downloaded():
    checked = 0
    for source, chunks in itertools.groupby(sorted(CHUNKS, key=lambda c: c.source),
                                            key=lambda c: c.source):  # fmt: skip
        if not cs.path_of(cs.BY_ID[source]).exists():
            continue
        text = cs.text(cs.BY_ID[source])
        for c in chunks:
            assert corpus.is_verbatim(c.text, text), c.id
            checked += 1
    if not checked:
        pytest.skip("no coaching source downloaded (tools/coaching_sources.py --download)")


def test_verbatim_ignores_pdf_artefacts_but_not_words():
    src = "the knee extensor muscu-\nlature is stretched and then shor tenﬁed"
    assert corpus.is_verbatim("knee extensor musculature is stretched", src)
    assert not corpus.is_verbatim("knee flexor musculature is stretched", src)


def test_runtime_needs_no_key_every_query_and_chunk_is_cached():
    emb = Embedder()
    assert emb.has([doc_input(c.text) for c in CHUNKS])
    assert emb.has([query_input(query_text(r.name, n)) for r in load_rules() for n in NEEDS])


def test_filter_keeps_the_fault_and_instep_or_general_kicks_only():
    got = filter_chunks(CHUNKS, "F-LEAN-BACK")
    assert got and all("F-LEAN-BACK" in c.faults and c.kick_type != "other" for c in got)
    assert "K-13" not in {c.id for c in got}  # tagged, but about a lofted pass


class FakeEmbedder:
    """Vectors by keyword, so the ranking is known."""

    def vectors(self, inputs):
        m = np.array([[("knee" in t) * 1.0, ("foot" in t) * 1.0, 0.1] for t in inputs])
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_rank_orders_by_similarity():
    chunks = [c for c in CHUNKS if c.id in ("K-02", "K-10", "K-35")]
    ranked = rank(chunks, "where does my foot go", FakeEmbedder())
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]
    assert ranked[0][0].id in ("K-02", "K-10")


@pytest.mark.parametrize("n,n_rel", [(1, 1), (3, 1), (5, 2), (6, 3), (8, 4)])
def test_expected_random_metrics_match_enumeration(n, n_rel):
    rel = set(range(n_rel))
    recalls, rrs = [], []
    for perm in itertools.permutations(range(n)):
        recalls.append(len(set(perm[:3]) & rel) / n_rel)
        rrs.append(next(1 / (i + 1) for i, x in enumerate(perm) if x in rel))
    recall, rr = expected_random_metrics(n, n_rel, n_rel, 3)
    assert recall == pytest.approx(mean(recalls)) and rr == pytest.approx(mean(rrs))


def test_relevance_labels_cover_every_rule_and_need_with_real_chunks():
    labels = yaml.safe_load(e6_retrieval.RELEVANCE.read_text())
    ids = {c.id for c in CHUNKS}
    for r in load_rules():
        assert set(labels[r.id]) == set(NEEDS)
        for need, rel in labels[r.id].items():
            assert set(rel) <= ids, (r.id, need)


def test_e6_reruns_without_a_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    rows = e6_retrieval.run()
    assert len(rows) == len(load_rules()) * len(NEEDS)
    md = e6_retrieval.report(rows, len(CHUNKS))
    committed = (e6_retrieval.RESULTS / "e6_summary.md").read_text()
    assert md == committed
