"""The coaching corpus's sources (Phase 8, E6): download them and print their text.

    uv run --group tools python tools/coaching_sources.py --download  # knowledge/coaching_sources/
    uv run --group tools python -I tools/coaching_sources.py --text <id>   # its text

Aamir approved this list on 2026-11-02 (DEC-037): open-access/openly licensed sources in
full, and short quoted excerpts only from the free-to-read copyrighted ones. Nothing here is
committed except this list; the chunks quoted from the sources are in
knowledge/coaching_corpus.yaml, each with its source's URL.
"""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = ROOT / "knowledge" / "coaching_sources"

SOURCES = [
    # -- already downloaded for the norms (knowledge/papers/, Phase 5) ---------------------
    {"id": "kellis_katis_2007", "licence": "open access (JSSM)", "group": "A",
     "url": "https://www.jssm.org/06-2-154.p_d_f", "paper": "kellis_katis_2007.pdf"},
    {"id": "scurr_hall_2009", "licence": "open access (JSSM)", "group": "A",
     "url": "https://www.jssm.org/08-2-230.p_d_f", "paper": "scurr_hall_2009.pdf"},
    {"id": "kapidzic_2014", "licence": "CC BY", "group": "A",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4234773/",
     "paper": "PMC4234773.bioc.json"},
    {"id": "xu_2025", "licence": "CC BY", "group": "A",
     "url": "https://doi.org/10.3389/fspor.2025.1605545", "paper": "xu_2025.pdf"},
    # -- openly licensed, new ----------------------------------------------------------------
    {"id": "shan_zhang_2011", "licence": "CC BY (BMC)", "group": "B",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3224572/",
     "fetch": "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
              "PMC3224572/unicode", "file": "PMC3224572.bioc.json"},
    # Approved but not used: the 2024 upper-body review (Springer) and wikiHow's "Kick a Ball"
    # both answered a script with a CDN bot-challenge page, and a bot check is not something
    # to get around (JOURNAL Phase 8).
    {"id": "wikipedia_shooting", "licence": "CC BY-SA 4.0", "group": "B",
     "url": "https://en.wikipedia.org/wiki/Shooting_(association_football)",
     "fetch": "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts|"
              "revisions&rvprop=ids|timestamp&explaintext=1&titles=Shooting_(association_football)",
     "file": "extract.json"},
    {"id": "plos_penalty_focus_2019", "licence": "CC BY 4.0 (PLOS ONE)", "group": "B",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6756529/",
     "fetch": "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
              "PMC6756529/unicode", "file": "PMC6756529.bioc.json"},
    # -- free to read, copyrighted: short quoted excerpts only -------------------------------
    {"id": "so_coaching_guide_2021", "licence": "© Special Olympics; excerpts only", "group": "C",
     "url": "https://media.specialolympics.org/resources/sports-essentials/coaching-guides/"
            "Sports-Essentials-Football-Coaching-Guide-2021.pdf", "file": "guide.pdf"},
    {"id": "so_wi_faults_fixes", "licence": "© Special Olympics Wisconsin; excerpts only",
     "group": "C",
     "url": "https://specialolympicswisconsin.org/wp-content/uploads/2024/10/"
            "Shooting-Progression-Faults-and-Fixes-Chart.pdf", "file": "chart.pdf"},
    {"id": "sonc_shooting_basics", "licence": "© Special Olympics North Carolina; excerpts only",
     "group": "C",
     "url": "https://sonc.org/wp-content/uploads/2025/06/Soccer-Shooting-Basics.pdf",
     "file": "basics.pdf"},
    {"id": "fa_finishing_2022", "licence": "© The FA; excerpts only", "group": "C",
     "url": "https://learn.englandfootball.com/articles/resources/2022/"
            "Different-finishing-techniques-in-football", "file": "page.html"},
]  # fmt: skip
BY_ID = {s["id"]: s for s in SOURCES}


def path_of(src: dict) -> Path:
    if "paper" in src:
        return ROOT / "knowledge" / "papers" / src["paper"]
    return SOURCES_DIR / src["id"] / src["file"]


def download(src: dict) -> None:
    import requests

    path = path_of(src)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(src.get("fetch", src["url"]), headers={"User-Agent": "Mozilla/5.0"},
                     timeout=60)  # fmt: skip
    r.raise_for_status()
    # A bot-challenge page comes back as 200 text/html; saved as paper.pdf, it looked like a
    # download (JOURNAL Phase 8). Check the body is what the file name promises.
    head = r.content.lstrip()[:4]
    if (path.suffix == ".pdf" and head != b"%PDF") or (
        path.suffix == ".json" and head[:1] not in (b"{", b"[")
    ):
        raise ValueError(f"expected {path.suffix}, got {r.headers.get('content-type')}")
    path.write_bytes(r.content)
    print(f"downloaded {src['id']}: {len(r.content)} bytes", flush=True)


class _Text(HTMLParser):
    """Visible text of an HTML page, one block per line; scripts and styles dropped."""

    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "div", "br", "tr", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _bioc_text(path: Path) -> str:
    doc = json.loads(path.read_text())[0]["documents"][0]
    return "\n\n".join(
        p["text"].strip()
        for p in doc["passages"]
        if p.get("text", "").strip() and p["infons"].get("type") != "ref"
    )


def text(src: dict) -> str:
    """The source's text with whitespace collapsed per line; PDFs keep [page n] markers."""
    path = path_of(src)
    if path.suffix == ".pdf":
        from pypdf import PdfReader

        raw = "\n\n".join(
            f"[page {i}]\n{p.extract_text()}" for i, p in enumerate(PdfReader(path).pages, 1)
        )
    elif path.name.endswith(".bioc.json"):
        raw = _bioc_text(path)
    elif path.suffix == ".json":
        pages = json.loads(path.read_text())["query"]["pages"]
        raw = next(iter(pages.values()))["extract"]
    else:
        p = _Text()
        p.feed(path.read_text(errors="replace"))
        raw = "".join(p.parts)
    lines = (re.sub(r"[ \t ]+", " ", ln).strip() for ln in raw.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--text", metavar="SOURCE_ID")
    args = ap.parse_args()
    if args.download:
        for src in SOURCES:
            try:
                download(src)
            except Exception as e:  # report every source, keep going
                print(f"FAILED {src['id']}: {type(e).__name__}: {e}", flush=True)
    if args.text:
        print(text(BY_ID[args.text]))


if __name__ == "__main__":
    main()
