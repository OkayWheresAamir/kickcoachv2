"""LLM-assisted norm extraction (handoff D12, S8): one structured-output Gemini call per paper,
then deterministic QA flags. Writes knowledge/norms_candidates.csv (never edited afterwards)
and, if it doesn't exist yet, knowledge/norms.csv as the copy Aamir verifies.

Papers are downloaded to knowledge/papers/ (gitignored; URLs in knowledge/sources.md). The
two "control" papers report their results only in figures, so every row extracted from them
is unsupported by construction: they measure the hallucinated-row rate (E5).

uv run --group tools python tools/extract_norms.py            # extract (cached calls)
uv run --group tools python tools/extract_norms.py --download # fetch missing papers first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from kickcoach.compare import norms
from kickcoach.compare.norms import VARIABLES, NormRow, Population, Stat
from kickcoach.llm import GeminiClient

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "knowledge" / "papers"
# The raw cache holds each paper's full text inside the prompt, so it stays local (gitignored,
# like the PDFs). RECORDS gets a redacted copy per paper: the answer, usage and a hash of the
# text, enough to audit the run without redistributing the papers.
CACHE = ROOT / "data" / "cache" / "norms_extraction"
RECORDS = ROOT / "results" / "norms_extraction"
# Strongest free-tier model not busy with E4 that day (pricing page, 2026-11-02: free input
# and output). Quotas are per model, so this leaves E4's and the feedback model's alone.
NORMS_MODEL = "gemini-3.6-flash"
PROMPT_VERSION = "norms-v1"

# page_offset: printed journal page = PDF page + offset (None: cite tables/sections only).
SOURCES = [
    {
        "id": "kellis_katis_2007",
        "role": "source",
        "citation": "Kellis & Katis (2007), J Sports Sci Med 6(2):154-165",
        "file": "kellis_katis_2007.pdf",
        "url": "https://www.jssm.org/06-2-154.p_d_f",
        "page_offset": 153,
    },
    {
        "id": "scurr_hall_2009",
        "role": "source",
        "citation": "Scurr & Hall (2009), J Sports Sci Med 8(2):230-234",
        "file": "scurr_hall_2009.pdf",
        "url": "https://www.jssm.org/08-2-230.p_d_f",
        "page_offset": 229,
    },
    {
        "id": "kapidzic_2014",
        "role": "source",
        "citation": "Kapidžić et al. (2014), J Hum Kinet 42:81-90",
        "file": "PMC4234773.bioc.json",
        "url": (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/"
            "PMC4234773/unicode"
        ),
        "page_offset": None,
    },
    {
        "id": "xu_2025",
        "role": "source",
        "citation": "Xu, Gong, Zhang & Zhang (2025), Front Sports Act Living 7:1605545",
        "file": "xu_2025.pdf",
        "url": (
            "https://www.frontiersin.org/journals/sports-and-active-living/articles/"
            "10.3389/fspor.2025.1605545/pdf"
        ),
        "page_offset": 0,
    },
    {
        "id": "nunome_ikegami_2006",
        "role": "control",
        "citation": "Nunome & Ikegami (2006), ISBS Proceedings, Salzburg",
        "file": "nunome_ikegami_2006.pdf",
        "url": "https://ojs.ub.uni-konstanz.de/cpa/article/download/135/96",
        "page_offset": 0,
    },
    {
        "id": "harrison_mannering_2006",
        "role": "control",
        "citation": "Harrison & Mannering (2006), ISBS Proceedings, Salzburg",
        "file": "harrison_mannering_2006.pdf",
        "url": "https://ojs.ub.uni-konstanz.de/cpa/article/download/247/206",
        "page_offset": 0,
    },
]

_NULLABLE_STR = {"type": ["string", "null"]}
_NULLABLE_NUM = {"type": ["number", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "enum": list(VARIABLES)},
                    "kick_type": {"type": "string"},
                    "condition": _NULLABLE_STR,
                    "population_level": _NULLABLE_STR,
                    "population_age_group": _NULLABLE_STR,
                    "population_sex": {
                        "type": "string",
                        "enum": ["male", "female", "mixed", "unspecified"],
                    },
                    "mean": _NULLABLE_NUM,
                    "sd": _NULLABLE_NUM,
                    "min": _NULLABLE_NUM,
                    "max": _NULLABLE_NUM,
                    "unit": {"type": "string"},
                    "n": {"type": ["integer", "null"]},
                    "method": {"type": "string"},
                    "angle_convention": {
                        "type": "string",
                        "enum": [
                            "interior_180_is_straight",
                            "flexion_0_is_straight",
                            "not_applicable",
                            "unclear",
                        ],
                    },
                    "page_or_table": {"type": "string"},
                    "primary_source": _NULLABLE_STR,
                    "quote": {"type": "string"},
                    "notes": _NULLABLE_STR,
                },
                "required": [
                    "variable", "kick_type", "condition", "population_level",
                    "population_age_group", "population_sex", "mean", "sd", "min", "max",
                    "unit", "n", "method", "angle_convention", "page_or_table",
                    "primary_source", "quote", "notes",
                ],
            },
        },
        "no_rows_reason": _NULLABLE_STR,
    },
    "required": ["rows", "no_rows_reason"],
}  # fmt: skip


def build_prompt(citation: str, text: str) -> str:
    variables = "\n".join(f"- {k}: {v}" for k, v in VARIABLES.items())
    return f"""\
You are extracting reference values (norms) from one biomechanics paper for an app that
measures football (soccer) instep kicks. The paper is: {citation}. Its full text follows,
with page or table markers in square brackets.

Variables of interest:
{variables}

Rules:
1. Extract only values printed as numbers in the text or in a table. Never read values off
   a figure or chart axis, never compute, average, convert units, or convert angle
   conventions. Copy every number exactly as printed.
2. One row per reported value: per variable, per group, per condition. If a table reports
   the same variable under several conditions (e.g. approach angles, kick types), give each
   condition its own row.
3. "quote" must be copied character for character from the text below and must contain the
   number(s) of the row. For a table, quote the table row as it appears (row label and
   values).
4. If the paper is a review and attributes the value to another study, put that study in
   "primary_source"; describe the population of that study if the paper says it.
5. "angle_convention": interior_180_is_straight if 180 deg is a straight knee;
   flexion_0_is_straight if 0 deg is a straight knee; unclear if the paper doesn't make it
   clear (explain in notes); not_applicable for distances and velocities.
6. "page_or_table": the page number from the markers, and the table number if any
   (e.g. "p. 232, Table 2").
7. "method": how the data were captured, as the paper states (e.g. "3D video (DLT)",
   "3D motion capture", "IMU", "2D video").
8. If the paper reports none of these variables as printed numbers, return no rows and say
   why in no_rows_reason.

Full text:
{text}
"""


def _pdf_text(path: Path, page_offset: int) -> str:
    from pypdf import PdfReader

    pages = []
    for i, page in enumerate(PdfReader(path).pages, 1):
        pages.append(f"[page {i + page_offset}]\n{page.extract_text()}")
    return "\n\n".join(pages)


def _bioc_text(path: Path) -> str:
    """PMC's BioC JSON: passages in order, tables numbered as they appear."""
    doc = json.loads(path.read_text())[0]["documents"][0]
    out, n_table = [], 0
    for p in doc["passages"]:
        kind = p["infons"].get("type", "")
        text = p.get("text", "").strip()
        if not text:
            continue
        if kind == "table_caption":
            n_table += 1
            out.append(f"[Table {n_table}] {text}")
        elif kind == "table":
            out.append(f"[Table {n_table}, body]\n{text}")
        elif kind != "ref":
            out.append(text)
    return "\n\n".join(out)


def source_text(src: dict) -> str:
    path = PAPERS / src["file"]
    if path.suffix == ".pdf":
        return _pdf_text(path, src["page_offset"])
    return _bioc_text(path)


def download(src: dict) -> None:
    import requests

    path = PAPERS / src["file"]
    if path.exists():
        return
    PAPERS.mkdir(parents=True, exist_ok=True)
    r = requests.get(src["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"downloaded {src['file']} ({len(r.content)} bytes)")


# -- deterministic QA (flags for the reviewer; they never change a row) -----------------------
def _norm(s: str) -> str:
    """Compare text the way a reader would: ignore layout whitespace, hyphenation at line
    ends, dash/minus variants and case. pypdf splits words ("rig ht hip"), so all
    whitespace goes. A table row rebuilt from separate columns still won't match: that flag
    means "check the table", not "invented"."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    # A word broken at a line end ("dis-\nplacement") reads the same as "dis- placement"
    # quoted from it; applied to both sides, so a dash before a space never decides a match.
    s = re.sub(r"-\s+", "", s)
    return re.sub(r"\s+", "", s).lower()


def _num_strings(x: float) -> set[str]:
    """Ways a number can be printed: 156.2, 156.20, .641, 1,092.23."""
    out = {f"{x:g}", f"{x:.1f}", f"{x:.2f}", f"{x:,.2f}", f"{x:,.1f}", f"{x:,.0f}"}
    out |= {s[1:] for s in list(out) if s.startswith("0.")}
    return {_norm(s) for s in out}


def qa_flags(row: dict, text: str) -> list[str]:
    flags = []
    quote = _norm(row["quote"])
    if not quote or quote not in _norm(text):
        flags.append("quote_not_found_verbatim")
    for k in ("mean", "sd", "min", "max"):
        v = row.get(k)
        if v is not None and not any(s in quote for s in _num_strings(v)):
            flags.append(f"{k}_not_in_quote")
    return flags


def write_record(src: dict, text: str, res: dict) -> None:
    usage = res["response"].get("usage_metadata", {})
    RECORDS.mkdir(parents=True, exist_ok=True)
    record = {
        "source_id": src["id"],
        "citation": src["citation"],
        "url": src["url"],
        "model": NORMS_MODEL,
        "prompt_version": PROMPT_VERSION,
        "cache_key": res["key"],
        "paper_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "paper_text_chars": len(text),
        "prompt_template": build_prompt(src["citation"], "<paper text, not stored>"),
        "text": res["text"],
        "parsed": res["parsed"],
        "usage": {k: usage.get(k) for k in ("prompt_token_count", "candidates_token_count",
                                             "thoughts_token_count", "total_token_count")},
    }  # fmt: skip
    (RECORDS / f"{src['id']}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))


def extract(sources: list[dict]) -> list[NormRow]:
    client = GeminiClient(NORMS_MODEL, CACHE)
    rows: list[NormRow] = []
    for src in sources:
        text = source_text(src)
        res = client.generate_json(
            prompt=build_prompt(src["citation"], text),
            schema=SCHEMA,
            temperature=None,
            tag=f"{PROMPT_VERSION} {src['id']}",
        )
        write_record(src, text, res)
        parsed = res["parsed"] or {"rows": [], "no_rows_reason": "unparseable response"}
        why = parsed["no_rows_reason"]
        print(
            f"{src['id']} ({src['role']}): {len(parsed['rows'])} rows"
            f"{' (cached)' if res['cached'] else ''}{'; no_rows_reason: ' + why if why else ''}",
            flush=True,
        )
        for r in parsed["rows"]:
            flags = qa_flags(r, text)
            note = r["notes"] or ""
            if flags:
                note = (note + " | " if note else "") + "QA: " + ", ".join(flags)
            rows.append(
                NormRow(
                    norm_id=f"N-{len(rows) + 1:03d}",
                    source_id=src["id"],
                    variable=r["variable"],
                    kick_type=r["kick_type"],
                    condition=r["condition"],
                    population=Population(
                        level=r["population_level"],
                        age_group=r["population_age_group"],
                        sex=r["population_sex"],
                    ),
                    stat=Stat(mean=r["mean"], sd=r["sd"], min=r["min"], max=r["max"]),
                    unit=r["unit"],
                    n=r["n"],
                    method=r["method"],
                    angle_convention=r["angle_convention"],
                    source=src["citation"],
                    page_or_table=r["page_or_table"],
                    primary_source=r["primary_source"],
                    quote=r["quote"],
                    notes=note or None,
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        for src in SOURCES:
            download(src)
    rows = extract(SOURCES)
    norms.save(rows, norms.CANDIDATES_CSV)
    print(f"{len(rows)} candidate rows -> {norms.CANDIDATES_CSV.relative_to(ROOT)}")
    if not norms.NORMS_CSV.exists():
        norms.save(rows, norms.NORMS_CSV)
        print(f"initialised {norms.NORMS_CSV.relative_to(ROOT)} for verification")


if __name__ == "__main__":
    main()
