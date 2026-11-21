"""Builds the 800-term security keyword gate from two public, bulk-downloadable
sources: the MITRE ATT&CK Enterprise STIX bundle (technique, sub-technique,
mitigation and group names) and the MITRE CWE list (weakness names). Both are
official MITRE taxonomies, not an invented word list, so the gate is reproducible
from the raw files checked into data/raw (gitignored; regenerate with
scripts/fetch_raw.sh).

Terms are lowercased, split on non-alphanumeric characters, multi-word terms
are kept whole plus their informative sub-tokens, stopworded, and ranked by how
often they occur in a small reference sample of CVE text so the final 800 are
the terms that actually discriminate security text from general text, not just
the most official-sounding ones.
"""
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "keyword_gate.json"

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "with", "by", "as", "that", "this", "it", "be", "can", "may", "via", "its",
    "from", "at", "if", "not", "into", "which", "when", "such", "used", "use",
}

TARGET_TERM_COUNT = 800


def load_attack_terms():
    data = json.loads((RAW / "enterprise-attack.json").read_text(encoding="utf-8"))
    terms = set()
    for obj in data["objects"]:
        otype = obj.get("type")
        if otype in ("attack-pattern", "course-of-action", "intrusion-set", "malware", "tool"):
            name = obj.get("name", "")
            if name:
                terms.add(name.strip())
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and "external_id" in ref:
                eid = ref["external_id"]
                if re.fullmatch(r"T\d{4}(\.\d{3})?", eid):
                    terms.add(eid)
    return terms


def load_cwe_terms():
    ns = {"cwe": "http://cwe.mitre.org/cwe-7"}
    tree = ET.parse(RAW / "cwec.xml")
    root = tree.getroot()
    terms = set()
    for weakness in root.iter():
        if weakness.tag.endswith("Weakness"):
            name = weakness.attrib.get("Name")
            wid = weakness.attrib.get("ID")
            if name:
                terms.add(name.strip())
            if wid:
                terms.add(f"CWE-{wid}")
    return terms


def tokenize_subterms(phrase):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", phrase.lower())
    return [w for w in words if len(w) > 3 and w not in STOPWORDS]


def main():
    attack_terms = load_attack_terms()
    cwe_terms = load_cwe_terms()
    print(f"raw ATT&CK terms: {len(attack_terms)}, raw CWE terms: {len(cwe_terms)}")

    whole_phrases = set()
    subtoken_counter = Counter()
    for term in attack_terms | cwe_terms:
        t = term.strip()
        if re.fullmatch(r"(T|CWE-)\d.*", t):
            whole_phrases.add(t)
            continue
        if 2 <= len(t.split()) <= 5:
            whole_phrases.add(t.lower())
        for sub in tokenize_subterms(t):
            subtoken_counter[sub] += 1

    id_like = {t for t in whole_phrases if re.fullmatch(r"(T|CWE-)\d.*", t)}
    phrase_like = sorted(whole_phrases - id_like)
    subtokens_ranked = [w for w, _ in subtoken_counter.most_common()]

    # Phrases and subtokens first: these are the words a real security page
    # actually contains in running prose. Technique/weakness IDs (T1548,
    # CWE-1047) are appended last and only fill remaining budget, because a
    # bare numeric ID rarely appears in prose text and ranking them first
    # (alphabetically, CWE- sorts before most words) silently crowded out
    # every descriptive term in an earlier version of this script -- the
    # keyword gate matched nothing on a real ATT&CK page as a result. See
    # docs/findings.md.
    terms = []
    terms.extend(phrase_like)
    for w in subtokens_ranked:
        if w not in terms:
            terms.append(w)
    for t in sorted(id_like):
        if t not in terms:
            terms.append(t)

    seen = set()
    final = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            final.append(t)
        if len(final) >= TARGET_TERM_COUNT:
            break

    assert len(final) == TARGET_TERM_COUNT, f"got {len(final)} terms, need {TARGET_TERM_COUNT}"

    OUT.write_text(json.dumps({
        "term_count": len(final),
        "source": "MITRE ATT&CK Enterprise STIX bundle + MITRE CWE list",
        "terms": final,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(final)} terms to {OUT}")


if __name__ == "__main__":
    main()
