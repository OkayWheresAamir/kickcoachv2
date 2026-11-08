"""Embeddings for the coaching corpus, cached in a committed .npz so retrieval runs without a
key (CLAUDE.md rule 5) and E6 reruns from a fresh clone.

The model is gemini-embedding-2 (free tier, the embeddings doc's recommended model on
2026-11-03; DEC-038). It takes no task_type: the task goes in a text prefix, one form for
documents and one for queries (embeddings doc). 768 dimensions is one of the doc's
recommended sizes; rows are normalised here anyway, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from kickcoach.llm import GeminiClient

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemini-embedding-2"
DIMS = 768
CACHE = ROOT / "knowledge" / "embeddings" / f"{MODEL}_{DIMS}.npz"
BATCH = 20  # inputs per request; the doc gives an 8192-token cap per request, not a count


def doc_input(text: str) -> str:
    # "title: none" rather than the source's name, so a chunk's location (e.g. a "drill"
    # column) can't leak into what E6 tests.
    return f"title: none | text: {text}"


def query_input(text: str) -> str:
    return f"task: search result | query: {text}"


def _key(text: str) -> str:
    return hashlib.sha256(f"{MODEL}|{DIMS}|{text}".encode()).hexdigest()[:24]


class Embedder:
    """``write_cache``: where new vectors go, if not ``cache``. The coach chat's free-text
    queries are open-ended, so they go to their own file (the eval's is committed, the app's
    is not) instead of growing the corpus file with every question."""

    def __init__(
        self, cache: Path = CACHE, client: GeminiClient | None = None,
        write_cache: Path | None = None,
    ) -> None:  # fmt: skip
        self.cache = Path(cache)
        self.write_cache = Path(write_cache) if write_cache else self.cache
        self._client = client
        self._vecs: dict[str, np.ndarray] = {}
        self._new: dict[str, np.ndarray] = {}
        for path in dict.fromkeys((self.cache, self.write_cache)):
            if path.exists():
                with np.load(path) as z:
                    self._vecs |= {k: z[k] for k in z.files}
                    if path == self.write_cache:
                        self._new = {k: z[k] for k in z.files}

    def vectors(self, inputs: list[str]) -> np.ndarray:
        """Unit vectors, one row per input (already prefixed with doc_input/query_input).
        Raises NoApiKey if any input is uncached and there is no key."""
        missing = list(dict.fromkeys(t for t in inputs if _key(t) not in self._vecs))
        if missing:
            client = self._client or GeminiClient(MODEL, self.cache.parent)
            for i in range(0, len(missing), BATCH):
                batch = missing[i : i + BATCH]
                for t, v in zip(batch, client.embed_texts(batch, DIMS), strict=True):
                    self._vecs[_key(t)] = self._new[_key(t)] = np.asarray(v, dtype=np.float32)
            self.write_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.write_cache, **self._new)
        m = np.stack([self._vecs[_key(t)] for t in inputs]).astype(np.float64)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def has(self, inputs: list[str]) -> bool:
        return all(_key(t) in self._vecs for t in inputs)
