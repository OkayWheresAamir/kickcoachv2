"""Retrieval over the coaching corpus (handoff D14): a metadata filter on the fault, then
embedding similarity to a query built from the fault rule and what the feedback needs.

The query is a fixed function of (fault rule, need), so the app's queries are a finite set
whose embeddings are committed with the corpus's: retrieval never needs a key at runtime.
"""

from __future__ import annotations

from math import comb

from kickcoach.retrieval.corpus import Chunk
from kickcoach.retrieval.embed import Embedder, doc_input, query_input

# What the feedback needs from a passage: the template's why_it_matters, cue and drill.
NEEDS = {
    "why": "Why does this matter for the kick?",
    "cue": "A short coaching cue: what should the player do instead?",
    "drill": "A practice drill to correct it.",
}
# The app measures instep kicks, so passages about other kicks are filtered out.
KICK_TYPES = ("instep", "general")


def query_text(fault_name: str, need: str) -> str:
    return f"{fault_name}. {NEEDS[need]}"


def filter_chunks(chunks: list[Chunk], fault_id: str) -> list[Chunk]:
    return [c for c in chunks if fault_id in c.faults and c.kick_type in KICK_TYPES]


def rank(chunks: list[Chunk], query: str, embedder: Embedder) -> list[tuple[Chunk, float]]:
    """Chunks by cosine similarity to the query, best first (ties broken by chunk id)."""
    if not chunks:
        return []
    docs = embedder.vectors([doc_input(c.text) for c in chunks])
    (q,) = embedder.vectors([query_input(query)])
    sims = docs @ q
    order = sorted(range(len(chunks)), key=lambda i: (-sims[i], chunks[i].id))
    return [(chunks[i], float(sims[i])) for i in order]


def retrieve(
    chunks: list[Chunk], fault_id: str, fault_name: str, need: str, embedder: Embedder,
    k: int = 3,
) -> list[tuple[Chunk, float]]:  # fmt: skip
    return rank(filter_chunks(chunks, fault_id), query_text(fault_name, need), embedder)[:k]


def expected_random_metrics(n: int, n_rel: int, n_rel_total: int, k: int) -> tuple[float, float]:
    """Recall@k and reciprocal rank expected when n candidates, n_rel of them relevant, come
    in a uniformly random order (the filter-only baseline has no ranking of its own).
    n_rel_total counts relevant chunks the filter dropped, which no order can recover."""
    if n_rel_total == 0:
        return float("nan"), float("nan")
    recall = (min(k, n) / n) * n_rel / n_rel_total if n else 0.0
    # P(first relevant at position r) = C(n - r, n_rel - 1) / C(n, n_rel)
    rr = 0.0
    if n_rel:
        rr = sum(comb(n - r, n_rel - 1) / comb(n, n_rel) / r for r in range(1, n - n_rel + 2))
    return recall, rr
