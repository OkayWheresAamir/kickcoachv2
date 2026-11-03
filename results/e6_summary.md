# E6: retrieval of coaching passages

Corpus: 55 verbatim chunks from 10 approved sources (`knowledge/coaching_corpus.yaml`). Embeddings: `gemini-embedding-2`, 768 dimensions, cosine similarity. Queries: one per fault rule and need (12), as the feedback step issues them. Relevance: `eval/e6_relevance.yaml`, labelled by Claude before any embedding was computed (DEC-038).

## Recall@3 and MRR (9 queries that have a relevant chunk)

| method | Recall@3 | MRR |
|---|---|---|
| filter only (expected over random order) | 0.46 | 0.61 |
| filter + embeddings | 0.56 | 0.75 |
| embeddings only (no filter) | 0.29 | 0.50 |

Query by query, filter + embeddings against filter only (MRR): better on 6, equal on 0, worse on 3; against embeddings only: better on 6, equal on 3, worse on 0.

## Per query

| fault | need | relevant | in filter | filtered | filter R@3 | filter+emb R@3 | filter+emb top 3 | emb R@3 | emb top 3 |
|---|---|---|---|---|---|---|---|---|---|
| F-LEAN-BACK | why | 3 | 2 | 3 | 0.67 | 0.67 | K-12 K-26 K-09 | 0.33 | K-43 K-12 K-32 |
| F-LEAN-BACK | cue | 1 | 1 | 3 | 1.00 | 1.00 | K-12 K-09 K-26 | 1.00 | K-08 K-12 K-01 |
| F-LEAN-BACK | drill | 0 | 0 | 3 | – | – | K-12 K-09 K-26 | – | K-08 K-12 K-09 |
| F-SHALLOW-BACKSWING | why | 3 | 3 | 8 | 0.38 | 0.67 | K-35 K-32 K-29 | 0.33 | K-35 K-32 K-42 |
| F-SHALLOW-BACKSWING | cue | 3 | 3 | 8 | 0.38 | 0.33 | K-35 K-02 K-32 | 0.00 | K-08 K-35 K-42 |
| F-SHALLOW-BACKSWING | drill | 0 | 0 | 8 | – | – | K-35 K-02 K-32 | – | K-35 K-42 K-08 |
| F-PLANT-FAR-BEHIND | why | 4 | 4 | 8 | 0.38 | 0.75 | K-40 K-38 K-39 | 0.50 | K-16 K-40 K-38 |
| F-PLANT-FAR-BEHIND | cue | 5 | 4 | 8 | 0.30 | 0.40 | K-09 K-38 K-02 | 0.20 | K-14 K-12 K-08 |
| F-PLANT-FAR-BEHIND | drill | 1 | 1 | 8 | 0.38 | 1.00 | K-09 K-10 K-38 | 0.00 | K-09 K-08 K-14 |
| F-REVERSED-SEQUENCE | why | 5 | 5 | 9 | 0.33 | 0.20 | K-34 K-21 K-29 | 0.20 | K-34 K-21 K-35 |
| F-REVERSED-SEQUENCE | cue | 1 | 1 | 9 | 0.33 | 0.00 | K-34 K-21 K-20 | 0.00 | K-08 K-09 K-34 |
| F-REVERSED-SEQUENCE | drill | 0 | 0 | 9 | – | – | K-34 K-21 K-20 | – | K-08 K-34 K-09 |

## Corpus gaps

3 of 12 queries have no relevant chunk at all: F-LEAN-BACK drill, F-SHALLOW-BACKSWING drill, F-REVERSED-SEQUENCE drill. No approved source gave a drill for those faults, so their drills stay uncited (the template's or the LLM's own wording).

## Caveats

- **Small n:** 9 scored queries; one query difference moves a mean by 0.11.
- **The labels are Claude's,** who also wrote the tags the filter uses. They were written from the chunk text with stated criteria before any ranking existed, but they are not independent.
- **Recall@3 has a ceiling below 1** for queries with more than 3 relevant chunks; every method shares it.
- **Filter-only has no ranking,** so a real app would order it somehow (e.g. by ID); the expectation over random orders is the neutral stand-in.
