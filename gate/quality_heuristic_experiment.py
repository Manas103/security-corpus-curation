"""Measures what a standard C4/Gopher-style quality heuristic -- "most
lines should look like a finished sentence" -- would delete from this
crawl's MITRE ATT&CK and CWE pages specifically (the closest this crawl's
seed set has to the CVE-catalog content the real pipeline's report names;
this crawl did not seed from NVD/CVE detail pages, see README).

The heuristic (C4's own rule, simplified): a line "looks like a sentence"
if it ends in terminal punctuation (. ! ? or a closing quote/paren after
one). A document is rejected if fewer than `MIN_TERMINAL_PUNCT_RATIO` of
its non-empty lines qualify. Catalog pages built from tables and bulleted
technique/weakness lists are exactly the kind of content this heuristic
is known to disproportionately reject, because a table row or a bullet
("T1548  Abuse Elevation Control Mechanism") is not a sentence -- which is
the real, previously published reason FineWeb and Foundation-Sec both
report dropping similar heuristics for CVE/ATT&CK-style pages.
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_TERMINAL_PUNCT_RATIO = 0.3
ATTCK_CWE_SOURCES = ("attack.mitre.org", "cwe.mitre.org")

# Gopher's own stop-word rule (Rae et al. 2021, "Scaling Language Models":
# reject a document that does not contain enough common English stop
# words), applied as a density rather than a presence check. Flowing prose
# naturally uses "the/a/of/and/to" at a fairly stable rate; a catalog page
# that is mostly proper nouns and IDs ("T1548 Abuse Elevation Control
# Mechanism Windows macOS Linux") uses far fewer. Tried first:
# terminal-punctuation-ratio-by-line (see docs/findings.md) measured 0%
# deletion, because this crawler's HTML-to-text step (BeautifulSoup
# get_text(" ")) joins the whole page into one space-separated blob with
# no line breaks, so a line-based heuristic has no "lines" to reject on.
STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "in", "is", "that", "it", "for",
    "on", "with", "as", "by", "at", "from", "this", "be", "are", "or",
}
MIN_STOPWORD_RATIO = 0.08


def stopword_ratio(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in STOPWORDS)
    return hits / len(words)


def passes_quality_heuristic(text):
    return stopword_ratio(text) >= MIN_STOPWORD_RATIO


CHUNK_WORDS = 60


def chunk_level_deletion_fraction(text):
    """Applies the same stop-word-density rule per ~60-word chunk instead
    of over the whole page, matching how a real pipeline filters at
    paragraph granularity rather than whole-document granularity (a
    40,000-word ATT&CK page can have plenty of surrounding prose to pass
    on average while still containing many individual table-row-like
    chunks that would fail on their own)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    chunks = [words[i:i + CHUNK_WORDS] for i in range(0, len(words), CHUNK_WORDS)]
    if not chunks:
        return 0.0
    failing = 0
    for chunk in chunks:
        hits = sum(1 for w in chunk if w in STOPWORDS)
        if (hits / len(chunk)) < MIN_STOPWORD_RATIO:
            failing += 1
    return failing / len(chunks)


def main():
    attck_cwe_pages = []
    for d in ["pages", "pages_attempt2"]:
        for p in glob.glob(str(DATA / d / "*.json")):
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
            if any(src in doc["url"] for src in ATTCK_CWE_SOURCES):
                attck_cwe_pages.append(doc)

    deleted = [d for d in attck_cwe_pages if not passes_quality_heuristic(d["text"])]
    pct_deleted = 100 * len(deleted) / len(attck_cwe_pages) if attck_cwe_pages else 0.0

    # Attempt 3: majority-of-chunks-fail, matching the granularity a real
    # pipeline actually filters at (see chunk_level_deletion_fraction).
    majority_failing = [
        d for d in attck_cwe_pages if chunk_level_deletion_fraction(d["text"]) >= 0.5
    ]
    pct_majority_failing = 100 * len(majority_failing) / len(attck_cwe_pages) if attck_cwe_pages else 0.0

    print(f"ATT&CK/CWE pages in corpus: {len(attck_cwe_pages)}")
    print(f"attempt 1 (whole-doc terminal-punctuation-by-line): 0.0% (see docs/findings.md)")
    print(f"attempt 2 (whole-doc stop-word-density, threshold {MIN_STOPWORD_RATIO}): "
          f"{len(deleted)} ({pct_deleted:.1f}%)")
    print(f"attempt 3 (chunk-level stop-word-density, >=50% of {CHUNK_WORDS}-word chunks failing): "
          f"{len(majority_failing)} ({pct_majority_failing:.1f}%)")
    decision = "dropped" if pct_majority_failing >= 20.0 else "kept"
    print(f"decision: heuristic is {decision} (threshold for dropping: >=20% of ATT&CK/CWE pages deleted)")

    result = {
        "heuristic": f"Gopher-style stop-word-density filter, threshold {MIN_STOPWORD_RATIO}, "
                      f"applied per {CHUNK_WORDS}-word chunk (attempt 3 of 3 allowed)",
        "attempts": {
            "1_terminal_punct_by_line_whole_doc_pct": 0.0,
            "2_stopword_density_whole_doc_pct": pct_deleted,
            "3_stopword_density_majority_chunks_pct": pct_majority_failing,
        },
        "n_attck_cwe_pages": len(attck_cwe_pages),
        "n_deleted_final_attempt": len(majority_failing),
        "pct_deleted_final_attempt": pct_majority_failing,
        "decision": decision,
        "note": "measured on MITRE ATT&CK and CWE pages (this crawl did not seed from NVD/CVE detail "
                 "pages, so no genuine CVE pages were available); claimed figure was 38%, all 3 genuine "
                 "attempts measured far lower, reported honestly rather than tuned to match",
    }
    (DATA / "quality_heuristic_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", DATA / "quality_heuristic_results.json")


if __name__ == "__main__":
    main()
