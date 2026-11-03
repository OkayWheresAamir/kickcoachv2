"""The coaching corpus (knowledge/coaching_corpus.yaml): load it, and check its quotes.

Each chunk is a verbatim excerpt. ``is_verbatim`` compares letters and digits only, so PDF
extraction artefacts (hyphenated line breaks, stray spaces, ligatures, list bullets'
punctuation) don't hide a real mismatch or fake one.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "knowledge" / "coaching_corpus.yaml"
KICK_TYPES = ("instep", "general", "other")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    where: str
    text: str
    faults: tuple[str, ...]
    kick_type: str
    level: str
    cite: str
    url: str


def load(path: Path = CORPUS) -> list[Chunk]:
    data = yaml.safe_load(path.read_text())
    srcs = data["sources"]
    return [
        Chunk(
            id=c["id"], source=c["source"], where=c["where"], text=c["text"],
            faults=tuple(c["faults"]), kick_type=c["kick_type"], level=c["level"],
            cite=srcs[c["source"]]["cite"], url=srcs[c["source"]]["url"],
        )
        for c in data["chunks"]
    ]  # fmt: skip


def _letters(s: str) -> str:
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s).lower())
    return re.sub(r"[^0-9a-z]", "", s.encode("ascii", "ignore").decode())


def is_verbatim(quote: str, source_text: str) -> bool:
    return _letters(quote) in _letters(source_text)
