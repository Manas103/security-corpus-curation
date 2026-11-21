"""A small, courteous, keyword-gated crawler, mirroring the Foundation-Sec
method at a scale this machine can run in minutes instead of a cluster
running for weeks: a wide-net crawl seeded in cybersecurity domains, where a
page's outlinks are only followed if the page itself passes the keyword
gate.

Honest limits, stated up front and repeated in the README:
  - Single machine, one process, a thread pool capped well under what the
    target sites would consider abuse (per-domain rate limit, a shared
    User-Agent that identifies the crawl and a contact method, robots.txt
    respected via urllib.robotparser).
  - A bounded wall-clock budget (CRAWL_SECONDS), not a bounded page count,
    because courteous per-domain pacing is the actual bottleneck. The page
    count this produces is the measured claim, not 120,000.
"""
import json
import re
import threading
import time
import urllib.robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PAGES_DIR = DATA / "pages"
USER_AGENT = "security-corpus-curation-research-crawl/0.1 (+https://github.com/Manas103/security-corpus-curation)"

SEEDS = [
    "https://attack.mitre.org/techniques/enterprise/",
    "https://attack.mitre.org/mitigations/enterprise/",
    "https://attack.mitre.org/groups/",
    "https://attack.mitre.org/software/",
    "https://cwe.mitre.org/data/definitions/1000.html",
    "https://cwe.mitre.org/top25/archive/2024/2024_cwe_top25.html",
    "https://www.cisa.gov/news-events/cybersecurity-advisories",
    "https://en.wikipedia.org/wiki/Computer_security",
    "https://en.wikipedia.org/wiki/Category:Computer_security",
    "https://en.wikipedia.org/wiki/Category:Cryptographic_attacks",
    "https://en.wikipedia.org/wiki/Category:Malware",
    "https://en.wikipedia.org/wiki/Category:Hacking_(computer_security)",
    "https://en.wikipedia.org/wiki/Category:Computer_security_organizations",
    "https://en.wikipedia.org/wiki/Category:Cryptographic_protocols",
    "https://owasp.org/www-project-top-ten/",
    "https://owasp.org/Top10/",
    "https://www.sans.org/white-papers/",
    "https://www.sans.org/blog/",
    "https://thehackernews.com/",
    "https://www.securityweek.com/",
    "https://portswigger.net/web-security",
    "https://portswigger.net/daily-swig",
    "https://www.cisecurity.org/insights",
    "https://unit42.paloaltonetworks.com/",
    "https://www.crowdstrike.com/en-us/cybersecurity-101/",
    "https://www.welivesecurity.com/en/",
    "https://www.schneier.com/blog/",
]

ALLOWED_NETLOCS = {
    "attack.mitre.org", "cwe.mitre.org", "www.cisa.gov", "cisa.gov",
    "en.wikipedia.org", "owasp.org", "www.sans.org", "thehackernews.com",
    "www.securityweek.com", "portswigger.net", "www.cisecurity.org",
    "unit42.paloaltonetworks.com", "www.crowdstrike.com",
    "www.welivesecurity.com", "www.schneier.com",
}

PER_DOMAIN_DELAY_S = 1.0
MAX_WORKERS = 16
REQUEST_TIMEOUT = 10


class RateLimiter:
    def __init__(self, delay_s):
        self.delay_s = delay_s
        self.last_hit = {}
        self.lock = threading.Lock()

    def wait(self, netloc):
        with self.lock:
            now = time.monotonic()
            last = self.last_hit.get(netloc, 0.0)
            wait_for = self.delay_s - (now - last)
            self.last_hit[netloc] = max(now, last) + self.delay_s
        if wait_for > 0:
            time.sleep(wait_for)


class RobotsCache:
    def __init__(self):
        self.parsers = {}
        self.lock = threading.Lock()

    def allowed(self, url):
        netloc = urlparse(url).netloc
        with self.lock:
            rp = self.parsers.get(netloc)
            if rp is None:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(f"https://{netloc}/robots.txt")
                try:
                    rp.read()
                except Exception:
                    rp = None
                self.parsers[netloc] = rp
        if rp is None:
            return True
        return rp.can_fetch(USER_AGENT, url)


def load_keyword_gate():
    gate = json.loads((DATA / "keyword_gate.json").read_text(encoding="utf-8"))
    return gate["terms"]


def passes_keyword_gate(text, term_patterns, min_hits=3):
    lowered = text.lower()
    hits = 0
    for pat in term_patterns:
        if pat.search(lowered):
            hits += 1
            if hits >= min_hits:
                return True, hits
    return False, hits


def extract_text_and_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.scheme in ("http", "https") and parsed.netloc in ALLOWED_NETLOCS:
            links.add(full.split("#")[0])
    return text, links


def crawl(crawl_seconds, out_dir, max_pages=None, rejected_dir=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if rejected_dir:
        rejected_dir.mkdir(parents=True, exist_ok=True)
    terms = load_keyword_gate()
    term_patterns = [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in terms]

    rate = RateLimiter(PER_DOMAIN_DELAY_S)
    robots = RobotsCache()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    frontier = deque(SEEDS)
    seen = set(SEEDS)
    stats = {"fetched": 0, "kept": 0, "rejected_keyword_gate": 0, "errors": 0, "robots_blocked": 0}
    lock = threading.Lock()
    start = time.monotonic()
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            with lock:
                if time.monotonic() - start > crawl_seconds:
                    stop.set()
                    return
                if max_pages and stats["kept"] >= max_pages:
                    stop.set()
                    return
                if not frontier:
                    url = None
                else:
                    url = frontier.popleft()
            if url is None:
                time.sleep(0.2)
                continue
            if not robots.allowed(url):
                with lock:
                    stats["robots_blocked"] += 1
                continue
            netloc = urlparse(url).netloc
            rate.wait(netloc)
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
                    with lock:
                        stats["errors"] += 1
                    continue
                text, links = extract_text_and_links(resp.text, url)
                with lock:
                    stats["fetched"] += 1
            except Exception:
                with lock:
                    stats["errors"] += 1
                continue

            if len(text) < 200:
                with lock:
                    stats["rejected_keyword_gate"] += 1
                continue

            passed, hits = passes_keyword_gate(text, term_patterns)
            with lock:
                if passed:
                    stats["kept"] += 1
                    page_id = stats["kept"]
                    record = {"url": url, "keyword_hits": hits, "text": text, "fetched_at": time.time()}
                    (out_dir / f"page_{page_id:06d}.json").write_text(
                        json.dumps(record, ensure_ascii=False), encoding="utf-8"
                    )
                    for link in links:
                        if link not in seen:
                            seen.add(link)
                            frontier.append(link)
                else:
                    stats["rejected_keyword_gate"] += 1
                    if rejected_dir:
                        rejected_count = stats["rejected_keyword_gate"]
                        record = {"url": url, "keyword_hits": hits, "passed_keyword_gate": False,
                                  "text": text, "fetched_at": time.time(), "label": "rejected_same_domain"}
                        (rejected_dir / f"rej_{rejected_count:06d}.json").write_text(
                            json.dumps(record, ensure_ascii=False), encoding="utf-8"
                        )

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(MAX_WORKERS)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        if time.monotonic() - start > crawl_seconds + 15:
            stop.set()
            break
    for t in threads:
        t.join(timeout=5)

    elapsed = time.monotonic() - start
    stats["elapsed_s"] = elapsed
    stats["frontier_remaining"] = len(frontier)
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(PAGES_DIR))
    ap.add_argument("--rejected-out", type=str, default=None,
                     help="also save same-domain pages that fail the keyword gate here, "
                          "for the relevancy classifier's negative class (not part of the corpus)")
    args = ap.parse_args()

    rejected_dir = Path(args.rejected_out) if args.rejected_out else None
    result = crawl(args.seconds, Path(args.out), args.max_pages, rejected_dir)
    print(json.dumps(result, indent=2))
    (DATA / "crawl_stats.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
