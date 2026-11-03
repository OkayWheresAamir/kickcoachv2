"""E6: does retrieval fetch the right coaching passage for a fault? (eval plan E6, handoff D14)

Queries are the ones the feedback step issues: one per fault rule and need (why it matters,
a cue, a drill). Relevance is eval/e6_relevance.yaml (Claude's labels, frozen before any
embedding was computed; DEC-038). Three methods:
1. **filter only:** chunks tagged with the fault (instep/general kicks). The filter has no
   order of its own, so its Recall@3 and MRR are the exact expectation over random orders.
2. **filter + embeddings:** the same chunks, ranked by cosine similarity to the query.
3. **embeddings only:** every chunk ranked, no filter (what Phase 9's free-text search faces).

    uv run python -m eval.e6_retrieval   # embeddings come from the committed cache

Writes results/e6.csv and results/e6_summary.md.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean

import yaml

from eval.e1_pose import table, write_csv
from kickcoach.compare.faults import load_rules
from kickcoach.retrieval import corpus
from kickcoach.retrieval.embed import MODEL, Embedder
from kickcoach.retrieval.retrieve import (
    NEEDS,
    expected_random_metrics,
    filter_chunks,
    query_text,
    rank,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RELEVANCE = Path(__file__).resolve().parent / "e6_relevance.yaml"
K = 3


def _metrics(ranked_ids: list[str], relevant: set[str]) -> tuple[float, float]:
    if not relevant:
        return float("nan"), float("nan")
    recall = len(set(ranked_ids[:K]) & relevant) / len(relevant)
    rr = next((1 / (i + 1) for i, c in enumerate(ranked_ids) if c in relevant), 0.0)
    return recall, rr


def run(embedder: Embedder | None = None) -> list[dict]:
    chunks = corpus.load()
    embedder = embedder or Embedder()
    labels = yaml.safe_load(RELEVANCE.read_text())
    rows = []
    for rule in load_rules():
        for need in NEEDS:
            relevant = set(labels[rule.id][need])
            query = query_text(rule.name, need)
            cand = filter_chunks(chunks, rule.id)
            in_filter = relevant & {c.id for c in cand}
            f_rec, f_rr = expected_random_metrics(len(cand), len(in_filter), len(relevant), K)
            fe = [c.id for c, _ in rank(cand, query, embedder)]
            e = [c.id for c, _ in rank(chunks, query, embedder)]
            fe_rec, fe_rr = _metrics(fe, relevant)
            e_rec, e_rr = _metrics(e, relevant)
            rows.append(
                {"fault_id": rule.id, "need": need, "query": query,
                 "n_relevant": len(relevant), "n_filtered": len(cand),
                 "n_relevant_in_filter": len(in_filter),
                 "filter_recall3": f_rec, "filter_rr": f_rr,
                 "filter_embed_recall3": fe_rec, "filter_embed_rr": fe_rr,
                 "embed_recall3": e_rec, "embed_rr": e_rr,
                 "filter_embed_top3": " ".join(fe[:K]), "embed_top3": " ".join(e[:K]),
                 "relevant": " ".join(sorted(relevant))}
            )  # fmt: skip
    return rows


def _f(x: float) -> str:
    return "–" if x != x else f"{x:.2f}"


def _wtl(diffs: list[float]) -> str:
    w = sum(d > 1e-9 for d in diffs)
    t = sum(abs(d) <= 1e-9 for d in diffs)
    return f"better on {w}, equal on {t}, worse on {len(diffs) - w - t}"


def report(rows: list[dict], n_chunks: int) -> str:
    scored = [r for r in rows if r["n_relevant"]]
    gaps = [r for r in rows if not r["n_relevant"]]
    methods = [
        ("filter only (expected over random order)", "filter"),
        ("filter + embeddings", "filter_embed"),
        ("embeddings only (no filter)", "embed"),
    ]
    L = [
        "# E6: retrieval of coaching passages",
        "",
        f"Corpus: {n_chunks} verbatim chunks from 10 approved sources "
        "(`knowledge/coaching_corpus.yaml`). Embeddings: "
        f"`{MODEL}`, 768 dimensions, cosine similarity. Queries: one per fault rule and need "
        f"({len(rows)}), as the feedback step issues them. Relevance: "
        "`eval/e6_relevance.yaml`, labelled by Claude before any embedding was computed "
        "(DEC-038).",
        "",
        f"## Recall@{K} and MRR ({len(scored)} queries that have a relevant chunk)",
        "",
        *table(
            ["method", f"Recall@{K}", "MRR"],
            [
                [
                    name,
                    _f(mean(r[f"{m}_recall3"] for r in scored)),
                    _f(mean(r[f"{m}_rr"] for r in scored)),
                ]
                for name, m in methods
            ],
        ),  # fmt: skip
        "",
        "Query by query, filter + embeddings against filter only (MRR): "
        + _wtl([r["filter_embed_rr"] - r["filter_rr"] for r in scored])
        + "; against embeddings only: "
        + _wtl([r["filter_embed_rr"] - r["embed_rr"] for r in scored])
        + ".",
        "",
        "## Per query",
        "",
        *table(
            [
                "fault",
                "need",
                "relevant",
                "in filter",
                "filtered",
                "filter R@3",
                "filter+emb R@3",
                "filter+emb top 3",
                "emb R@3",
                "emb top 3",
            ],
            [
                [
                    r["fault_id"],
                    r["need"],
                    str(r["n_relevant"]),
                    str(r["n_relevant_in_filter"]),
                    str(r["n_filtered"]),
                    _f(r["filter_recall3"]),
                    _f(r["filter_embed_recall3"]),
                    r["filter_embed_top3"],
                    _f(r["embed_recall3"]),
                    r["embed_top3"],
                ]
                for r in rows
            ],
        ),  # fmt: skip
        "",
        "## Corpus gaps",
        "",
        f"{len(gaps)} of {len(rows)} queries have no relevant chunk at all: "
        + ", ".join(f"{r['fault_id']} {r['need']}" for r in gaps)
        + ". No approved source gave a drill for those faults, so their drills stay "
        "uncited (the template's or the LLM's own wording).",
        "",
        "## Caveats",
        "",
        f"- **Small n:** {len(scored)} scored queries; one query difference moves a mean by "
        f"{1 / len(scored):.2f}.",
        "- **The labels are Claude's,** who also wrote the tags the filter uses. They were "
        "written from the chunk text with stated criteria before any ranking existed, but "
        "they are not independent.",
        f"- **Recall@{K} has a ceiling below 1** for queries with more than {K} relevant "
        "chunks; every method shares it.",
        "- **Filter-only has no ranking,** so a real app would order it somehow (e.g. by ID); "
        "the expectation over random orders is the neutral stand-in.",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    rows = run()
    write_csv(RESULTS / "e6.csv", rows)
    md = report(rows, len(corpus.load()))
    (RESULTS / "e6_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
