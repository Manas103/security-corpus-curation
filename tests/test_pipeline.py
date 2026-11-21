"""Offline unit tests: keyword gate construction and matching, and the
teacher-labeling heuristic the classifier is distilled from. No network
access and no model downloads -- the crawler, classifier training and
continued-pretraining scripts are integration-level and run manually (see
README); this suite covers the parts of the pipeline that are pure
functions over text.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "crawler"))
sys.path.insert(0, str(ROOT / "classifier"))

from crawl import extract_text_and_links, load_keyword_gate, passes_keyword_gate  # noqa: E402
from train_classifier import ANCHOR_TERMS, teacher_label  # noqa: E402


def test_keyword_gate_has_800_terms():
    gate = json.loads((ROOT / "data" / "keyword_gate.json").read_text(encoding="utf-8"))
    assert gate["term_count"] == 800
    assert len(gate["terms"]) == 800
    assert len(set(gate["terms"])) == 800  # no duplicates


def test_keyword_gate_terms_are_lowercase_and_nonempty():
    terms = load_keyword_gate()
    for t in terms:
        assert t == t.lower()
        assert len(t.strip()) > 0


def _patterns():
    return [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in load_keyword_gate()]


def test_passes_keyword_gate_on_real_security_text():
    patterns = _patterns()
    terms = load_keyword_gate()
    # Build a sentence out of 4 real gate terms rather than hand-written
    # prose, so the test does not depend on phrasing matching the gate's
    # exact vocabulary by luck (see docs/findings.md: an earlier version of
    # this gate matched almost nothing on real pages for exactly this
    # reason, just inverted -- the gate's own terms not appearing in it).
    sample_terms = [t for t in terms if " " not in t or len(t.split()) <= 3][:4]
    text = "This report discusses " + ", ".join(sample_terms) + " in detail."
    passed, hits = passes_keyword_gate(text, patterns, min_hits=3)
    assert passed
    assert hits >= 3


def test_passes_keyword_gate_rejects_unrelated_text():
    patterns = _patterns()
    text = "The chef kneaded the sourdough for ten minutes before letting it rest overnight in a warm kitchen."
    passed, hits = passes_keyword_gate(text, patterns, min_hits=3)
    assert not passed
    assert hits < 3


def test_extract_text_and_links_strips_script_and_nav():
    html = """
    <html><body>
      <nav>Home | About</nav>
      <script>var x = 1;</script>
      <p>Real content about malware analysis.</p>
    </body></html>
    """
    text, links = extract_text_and_links(html, "https://example.com/")
    assert "Real content about malware analysis" in text
    assert "var x" not in text
    assert "Home" not in text


def test_teacher_label_independent_of_keyword_gate_vocabulary():
    gate_terms = set(t.lower() for t in load_keyword_gate())
    assert not (set(ANCHOR_TERMS) & gate_terms), "anchor terms must stay disjoint from the keyword gate"


def test_teacher_label_positive_and_negative():
    positive_text = "This cyberattack caused a major data breach after a successful phishing campaign."
    negative_text = "The orchestra performed a beautiful sonata to a full concert hall on Friday evening."
    assert teacher_label(positive_text) == 1
    assert teacher_label(negative_text) == 0
