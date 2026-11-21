"""Fetches a small set of deliberately off-topic Wikipedia pages to serve as
hard-ish negatives for the relevancy classifier (held out from the security
corpus itself; never counted toward the "crawled cybersecurity pages" claim).

Without genuine negatives, "relevancy classifier" has nothing to discriminate
against: every page this project's own crawler keeps already passed the
800-term keyword gate by construction (crawl.py only saves pages that pass
it), so training a classifier only on those would measure nothing. Real
negatives close that gap honestly instead of reusing the keyword gate itself
as ground truth, which would make the keyword-alone baseline trivially 1.0.
"""
import json
import time
from pathlib import Path

import requests

from crawl import USER_AGENT, extract_text_and_links, load_keyword_gate, passes_keyword_gate
import re

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "negatives"

TOPICS = [
    "Pasta", "Baking", "Sourdough", "Risotto", "Tapas", "Sushi",
    "Figure_skating", "Marathon", "Cricket", "Badminton", "Snooker", "Darts",
    "Watercolor_painting", "Opera", "Ballet", "Jazz", "Baroque_music", "Origami",
    "Mount_Everest", "Great_Barrier_Reef", "Sahara", "Amazon_rainforest",
    "Volcano", "Glacier", "Coral_reef", "Monsoon",
    "Gothic_architecture", "Bridge_(structure)", "Lighthouse", "Greenhouse",
    "Beekeeping", "Viticulture", "Dairy_farming", "Crop_rotation",
    "Chess", "Sudoku", "Crossword", "Jigsaw_puzzle",
    "Tango", "Flamenco", "Bagpipes", "Harpsichord",
    "Tea_ceremony", "Calligraphy", "Bonsai", "Ikebana",
    "Rugby_union", "Table_tennis", "Archery", "Fencing",
    "Astronomy", "Meteorology", "Oceanography", "Paleontology",
    "Butterfly", "Hedgehog", "Owl", "Dolphin",
    "Vineyard", "Cheese", "Chocolate", "Coffee",
    "Gardening", "Carpentry", "Pottery", "Weaving",
    "Sailing", "Kayaking", "Mountaineering", "Birdwatching",
    "Poetry", "Sonnet", "Haiku", "Mythology",
    "Carnival", "Festival", "Parade", "Folk_dance",
]


def fetch_negatives():
    OUT.mkdir(parents=True, exist_ok=True)
    terms = load_keyword_gate()
    patterns = [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in terms]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    kept = 0
    for topic in TOPICS:
        url = f"https://en.wikipedia.org/wiki/{topic}"
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            text, _ = extract_text_and_links(resp.text, url)
            if len(text) < 200:
                continue
            passed, hits = passes_keyword_gate(text, patterns)
            kept += 1
            record = {
                "url": url, "keyword_hits": hits, "passed_keyword_gate": passed,
                "text": text, "fetched_at": time.time(), "label": "negative_topic",
            }
            (OUT / f"neg_{kept:04d}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        except Exception:
            continue
        time.sleep(0.5)
    print(f"fetched {kept} negative-topic pages")
    return kept


if __name__ == "__main__":
    fetch_negatives()
