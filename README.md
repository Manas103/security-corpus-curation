# Security-Corpus Curation with a Token-Budget-Matched Pretraining Ablation

Reproduces, at a scale one machine can run in an afternoon instead of a
cluster running for weeks, the data-curation pipeline a published security
LLM's technical report describes: a courteous keyword-gated crawler seeded
in cybersecurity domains, an 800-term keyword gate, a small classifier
distilled from a heuristic teacher's relevancy labels, and a token-budget-
matched continued-pretraining ablation that asks whether the extra curation
stage is worth anything once compute is held fixed. Every number below was
measured on this machine, not targeted; several are far short of the scale
a real pipeline like this would run at, and that gap is stated plainly
rather than hidden.

Python 3.12, PyTorch 2.6 (CUDA 12.4), Transformers, scikit-learn, requests
and BeautifulSoup for the crawler.

## Honest framing

- **This is a few million tokens, not 5.1 billion.** The published pipeline
  this reproduces (Foundation-Sec) crawled over 4 TiB raw down to 5.1B
  tokens. This project's crawl ran for about 40 minutes total across two
  attempts, courteously rate-limited (1 request/second/domain), and
  produced a few thousand pages. The *structure* of the experiment --
  keyword gate, distilled classifier, equal-token-budget ablation -- is
  what is being reproduced, not the scale.
- **The relevancy classifier's ground truth is a heuristic, not a frontier
  model.** The real pipeline paid a frontier LLM to label 26,000 documents.
  This project substitutes a disjoint 24-term anchor-phrase heuristic
  (`classifier/train_classifier.py`), deliberately built from different
  vocabulary than the 800-term keyword gate so the keyword-alone baseline
  is not circular. It is a weaker, noisier teacher than a frontier model
  would be; the comparison's qualitative shape (classifier beats keywords
  alone) is the thing being tested, not the absolute F1 values.
  "Distilled" here means distilled from that heuristic teacher's labels
  into a small model, not architecturally distilled from a transformer:
  given the corpus's scale, the classifier is TF-IDF + logistic
  regression, not a small transformer.
- **The crawl only saves pages that already pass the keyword gate.**
  "Crawled public cybersecurity pages" below counts gate-passing pages;
  pages fetched and rejected are counted separately (see Results) and
  were not kept, matching the pipeline's own two-stage filter, not a
  third, broader corpus.
- **The corpus is almost entirely MITRE ATT&CK and CWE, not the broader
  mix of sources seeded.** 14 seed domains were configured (OWASP, SANS,
  SecurityWeek, Krebs-style outlets, Wikipedia categories, etc.), but the
  keyword-gated BFS crawl's frontier was dominated by ATT&CK's and CWE's
  own dense internal cross-linking (a technique page links to dozens of
  other technique/mitigation/group pages on the same site); 1,489 of
  1,490 kept pages came from `attack.mitre.org` or `cwe.mitre.org`. See
  Findings.
- **The continued-pretraining token budget is set by the smaller of the
  two filtered corpora, not a target.** Equal-token-budget is the point
  of the experiment; the actual number is whatever both arms can supply
  (see Results), which is far below the 60M tokens a larger crawl would
  allow.
- **Held-out "domain accuracy" is a multiple-choice cloze task, not a
  standard benchmark.** A held-out document's sentence containing a
  security term has that term swapped for one of 3 random decoys; the
  model's completion loss on each of the 4 candidates picks a winner.
  Accuracy is how often the correct term wins. This is a defensible but
  improvised proxy for "did continued pretraining teach the model
  anything about this domain," not a published eval.
- **Machine:** Windows 11, RTX 4070 SUPER (12 GB), CUDA 12.4, PyTorch
  2.6.0+cu124, Qwen2.5-0.5B (494M parameters) as the base model.

## Why this exists

A small version of the data half of pretraining a domain-specialized LLM:
decide what to keep before a single GPU-hour of pretraining is spent on
it, and prove the decision was worth something by holding compute fixed
and measuring the result, not just asserting "more curated data is
better."

## Architecture

```
gate/build_keyword_gate.py     builds the 800-term gate from real MITRE ATT&CK + CWE data
crawler/crawl.py                 courteous, robots-respecting, keyword-gated BFS crawler
crawler/fetch_negatives.py       off-topic Wikipedia pages: negatives for the classifier
classifier/train_classifier.py   heuristic teacher labels; TF-IDF+LR "distilled" classifier
pretrain/continued_pretrain.py    token-budget-matched continued-pretraining ablation, 3 seeds
tests/test_pipeline.py           offline unit tests (no network, no model download)
data/                             keyword_gate.json, crawled pages, classifier + ablation results (gitignored except summaries)
```

Design notes:

- **The keyword gate is built from real taxonomies, not an invented word
  list.** `gate/build_keyword_gate.py` downloads the MITRE ATT&CK
  Enterprise STIX bundle and the MITRE CWE list and derives 800 terms from
  their technique/weakness names and IDs, so the gate is reproducible from
  public data rather than hand-picked.
- **The crawler mirrors the published method's dependency order:** a page's
  outlinks are only followed if the page itself passes the keyword gate,
  so the frontier narrows toward security content over time instead of
  wandering the general web.
- **Per-domain rate limiting, not a global one.** Courtesy is about not
  hammering any one server, not about overall throughput; 16 worker
  threads can be fetching 16 different domains at 1 req/s each without
  being rude to any of them.

## Validation

`tests/test_pipeline.py` (offline, no network): the keyword gate has
exactly 800 unique lowercase terms; the gate matches real security
prose and rejects unrelated prose; HTML extraction strips nav/script
content; the classifier's anchor-term teacher vocabulary is disjoint
from the keyword gate's vocabulary (the thing that keeps the
keyword-alone baseline from being circular); the teacher labels a
security paragraph positive and an unrelated one negative.

## Findings

**An early version of the keyword gate matched almost nothing on real
pages, because the term list was built in the wrong order.** The gate
builder combined ATT&CK technique names/IDs and CWE weakness names/IDs,
then filled 800 slots by iterating: numeric IDs first (sorted
alphabetically), then phrases, then sub-tokens. There are more CWE IDs
alone (1,968) than the 800-term budget, so the first version's gate was
entirely "CWE-1004", "CWE-1047", etc. -- terms that essentially never
appear as literal text in a real page's prose. Running the gate against
the actual seed page `attack.mitre.org/techniques/enterprise/` (304,947
characters of real ATT&CK technique descriptions) found **0 of 800 terms**
anywhere in it. The fix: rank phrases and sub-tokens (the words a real
page's prose actually contains) first, and only fill remaining budget
with bare IDs last. After the fix, the same page matches on real
vocabulary ("abuse elevation control mechanism", "accessibility
features", etc.) within the first few hundred words. See
`gate/build_keyword_gate.py`'s comments for the corrected ordering.

**The crawl's frontier collapsed onto two domains instead of diversifying
across the 14 seeded.** 14 seed domains were configured specifically to
mirror a wide-net crawl, but a keyword-gated BFS crawler follows whichever
outlinks a kept page actually has, and MITRE's own site structure links
each ATT&CK technique page to dozens of sibling technique, mitigation,
group and software pages. That internal link density means the frontier
queue filled with `attack.mitre.org` and `cwe.mitre.org` URLs far faster
than it could reach the single-page seeds for the other 12 domains before
the time budget ran out: 1,489 of 1,490 kept pages (99.9%) came from
those two sites. A courteous crawl with a short wall-clock budget
structurally favors whichever seed has the densest same-domain link graph,
not necessarily the most sources -- a real wide-net crawl would need
either a much larger time budget or an explicit per-domain page quota to
actually diversify.

**The "one quality heuristic deletes a large fraction of catalog pages"
claim needed 3 attempts, and even the last one measured far below the
claimed 38%.** Attempt 1 (a C4-style rule: most lines should end in
terminal punctuation) measured exactly 0% deletion, because this
crawler's HTML-to-text extraction (`BeautifulSoup.get_text(" ")`) joins
an entire page into one space-separated blob with no line breaks, leaving
a line-based heuristic nothing to reject on. Attempt 2 (a Gopher-style
stop-word-density rule, applied over the whole document) measured 0.1%:
whole ATT&CK pages contain enough surrounding descriptive prose
(technique descriptions, mitigation notes, detection guidance) to keep
document-level stop-word density normal even though individual table rows
and bullet lists inside the same page would fail on their own. Attempt 3
(the same stop-word rule applied per 60-word chunk, rejecting a page only
if most of its chunks fail) measured 0.8%. All 3 are genuine, honestly
reported attempts at the same claim, not 3 different claims; the heuristic
this project actually tested would be **kept**, not dropped, which
disagrees with the resume's claim. See `data/quality_heuristic_results.json`.

## Measured results

Machine: Windows 11, RTX 4070 SUPER, CUDA 12.4, PyTorch 2.6.0+cu124.

| What | Measured |
|---|---|
| Keyword gate | 800 unique terms, built from MITRE ATT&CK (STIX) + MITRE CWE |
| Pages crawled and kept (passed the gate) | see build_report.json for the exact count across 2 crawl attempts |
| Pages fetched from security-seeded domains but rejected by the gate | reported alongside the kept count |
| Relevancy classifier held-out F1 | **0.878** (1,404 documents, 811 teacher-positive; 351 held out) |
| Keyword-gate-alone held-out F1 (same held-out set, same teacher) | **0.759** |
| Continued-pretraining equal-token-budget ablation | token budget set honestly (3,471,360 tokens/arm, the smaller filtered corpus's size); the domain-accuracy delta itself did not finish training within the session's available time (see Limitations and `docs/pretrain_output.txt`) |

Full raw output under `docs/`.

## Building and running

```bash
python -m venv venv && venv/Scripts/python -m pip install -r requirements.txt
python gate/build_keyword_gate.py                       # needs data/raw/{enterprise-attack.json,cwec.xml}
python crawler/crawl.py --seconds 900 --out data/pages   # courteous; runs for the given wall-clock budget
python crawler/fetch_negatives.py                        # off-topic negatives for the classifier
python classifier/train_classifier.py                    # writes data/classifier_results.json + .pkl
python pretrain/continued_pretrain.py                     # needs a CUDA GPU; writes data/pretrain_ablation_results.json
pytest tests -q                                           # offline, no network or GPU needed
```

## Limitations

- **The continued-pretraining ablation did not finish.** Qwen2.5-0.5B's
  151,936-token vocabulary makes the logits/cross-entropy step far more
  expensive than its 494M parameter count suggests; the first run (fp32)
  ran at 100% GPU utilization for over 25 minutes without completing even
  one of 6 planned training runs. A faster configuration (bf16, 1 epoch,
  a capped token budget) is in `pretrain/continued_pretrain.py` but was
  not re-run within this session. The domain-accuracy-delta and
  3-seed-spread claims are reported as not measured, not estimated.
- The crawl is a few thousand pages against a stated claim of a much
  larger corpus; see Honest framing.
- The relevancy classifier's ground truth is a disclosed heuristic
  substitute for frontier-model labeling, not the real thing.
- The continued-pretraining token budget is far below what a production
  ablation would use, and "domain accuracy" is an improvised cloze eval,
  not a published benchmark.
- No deduplication is applied to the crawled corpus itself (that is the
  subject of the companion repo, `security-corpus-contamination-audit`).
