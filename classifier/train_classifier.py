"""Trains a small text classifier distilled from a heuristic teacher's
relevancy labels, and compares it against the 800-term keyword gate alone,
on a held-out split -- the same two-number comparison Foundation-Sec reports
(F1 0.924 for the classifier, implicitly lower for keywords alone) [see
README "Honest framing" for why a heuristic teacher substitutes for their
frontier-model-labeled 26,000 documents here].

Teacher ground truth, by design independent of the 800-term keyword gate
(gate/keyword_gate.json): a document is relevant iff it contains at least
2 hits from a disjoint 24-term anchor list of plain-English security
vocabulary that is NOT in the CWE/ATT&CK-derived gate. This keeps the
keyword-alone baseline from being circular (the gate and the teacher must
disagree sometimes, or "keywords alone" would trivially score 1.0 against
a teacher built from the same words).
"""
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "crawler"))
from crawl import load_keyword_gate  # noqa: E402

ANCHOR_TERMS = [
    "cyberattack", "data breach", "zero-day", "ransomware", "phishing",
    "malicious actor", "threat intelligence", "incident response",
    "security vulnerability", "penetration test", "security researcher",
    "exploit code", "patch tuesday", "threat actor", "cybercriminal",
    "security flaw", "malicious code", "unauthorized access",
    "security breach", "cyber threat", "vulnerability disclosure",
    "attack surface", "security researcher", "compromised",
]
ANCHOR_PATTERNS = [re.compile(r"\b" + re.escape(t) + r"\b") for t in ANCHOR_TERMS]


def teacher_label(text):
    lowered = text.lower()
    hits = sum(1 for p in ANCHOR_PATTERNS if p.search(lowered))
    return 1 if hits >= 2 else 0


def load_corpus():
    gate_terms = load_keyword_gate()
    gate_patterns = [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in gate_terms]

    texts, teacher_labels, gate_preds, sources = [], [], [], []
    pos_files = []
    for d in ["pages", "pages_attempt2"]:
        pos_files += glob.glob(str(DATA / d / "*.json"))
    neg_files = glob.glob(str(DATA / "negatives" / "*.json"))

    for path in pos_files + neg_files:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        text = doc["text"][:20000]  # cap pathologically long catalog pages (one ATT&CK page is 300K+ chars)
        texts.append(text)
        teacher_labels.append(teacher_label(text))
        lowered = text.lower()
        hits = sum(1 for p in gate_patterns if p.search(lowered))
        gate_preds.append(1 if hits >= 3 else 0)
        sources.append("corpus" if "negatives" not in path else "negative_topic")

    return texts, np.array(teacher_labels), np.array(gate_preds), sources


def main():
    texts, y, gate_pred, sources = load_corpus()
    n_pos = int(y.sum())
    print(f"loaded {len(texts)} documents ({n_pos} teacher-positive, {len(texts) - n_pos} teacher-negative)")

    idx = np.arange(len(texts))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=0, stratify=y)

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    gate_pred_test = gate_pred[test_idx]

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 1), min_df=2, max_df=0.9, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    clf_pred_test = clf.predict(X_test)

    f1_classifier = f1_score(y_test, clf_pred_test)
    f1_keywords_alone = f1_score(y_test, gate_pred_test)

    print(f"held-out set: {len(test_idx)} documents ({int(y_test.sum())} teacher-positive)")
    print(f"keyword-gate-alone held-out F1: {f1_keywords_alone:.4f}")
    print(f"distilled classifier held-out F1: {f1_classifier:.4f}")

    result = {
        "teacher": "disjoint 24-term anchor-phrase heuristic (>=2 hits), substitute for frontier-model labeling",
        "classifier": "TF-IDF (4000 features, 1-2 grams) + L2 logistic regression, class_weight=balanced",
        "n_documents": len(texts),
        "n_teacher_positive": n_pos,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_test_positive": int(y_test.sum()),
        "f1_keywords_alone": f1_keywords_alone,
        "f1_classifier": f1_classifier,
    }
    (DATA / "classifier_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(DATA / "classifier_sklearn.pkl", "wb") as f:
        pickle.dump((vectorizer, clf), f)
    print("wrote", DATA / "classifier_results.json", "and classifier_sklearn.pkl")


if __name__ == "__main__":
    main()
