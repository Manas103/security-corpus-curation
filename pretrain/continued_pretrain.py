"""Token-budget-matched continued-pretraining ablation: take two corpora
built from the same crawl by two different filters (the 800-term keyword
gate alone, and the keyword gate followed by the distilled relevancy
classifier), truncate both to the identical token budget, continued-
pretrain a copy of the same 0.5B base model on each, and measure held-out
domain accuracy. The equal-token constraint is the entire point: without
it, "classifier filter wins" would just mean "classifier filter kept
different documents," not that curation quality moved anything at a fixed
compute cost.

Honest scale: the real pipeline this reproduces (Foundation-Sec) used a
5.1B-token corpus and presumably a much larger budget per ablation arm.
This project's crawl is a few million tokens (see README), so the token
budget here is the largest equal amount both filtered corpora can supply,
not 60M -- the structure of the experiment is what is being reproduced at
small scale, not the absolute numbers.
"""
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "crawler"))
sys.path.insert(0, str(ROOT / "classifier"))
from crawl import load_keyword_gate  # noqa: E402
from train_classifier import ANCHOR_TERMS, teacher_label  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BLOCK_SIZE = 512
SEEDS = [0, 1, 2]
EVAL_HOLD_OUT_FRAC = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Qwen2.5-0.5B's vocab is 151,936 -- the logits/cross-entropy over that
# vocab, not the 0.5B parameters themselves, dominate step time and VRAM
# (first attempt measured ~12 GB used in fp32, close to this GPU's 12 GB
# ceiling, and one epoch over the natural token budget did not finish in a
# practical time; see docs/findings.md). bf16 (Ada supports it natively)
# roughly halves both; MAX_TOKEN_BUDGET caps wall-clock time explicitly
# rather than silently, on top of the natural equal-budget constraint.
TRAIN_DTYPE = torch.bfloat16
MAX_TOKEN_BUDGET = 1_200_000


def load_all_docs():
    docs = []
    for d in ["pages", "pages_attempt2"]:
        for p in glob.glob(str(DATA / d / "*.json")):
            doc = json.loads(Path(p).read_text(encoding="utf-8"))
            docs.append(doc["text"])
    return docs


def build_filters(docs, classifier_path):
    import pickle
    with open(classifier_path, "rb") as f:
        vectorizer, clf = pickle.load(f)

    keyword_filtered = docs  # every stored page already passed the keyword gate (see crawler/crawl.py)
    X = vectorizer.transform(docs)
    classifier_mask = clf.predict(X).astype(bool)
    two_stage_filtered = [d for d, keep in zip(docs, classifier_mask) if keep]
    return keyword_filtered, two_stage_filtered


def tokenize_and_pack(tokenizer, texts, max_tokens):
    ids = []
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        ids.append(tokenizer.eos_token_id)
        if len(ids) >= max_tokens:
            break
    ids = ids[:max_tokens]
    n_blocks = len(ids) // BLOCK_SIZE
    ids = ids[: n_blocks * BLOCK_SIZE]
    blocks = torch.tensor(ids, dtype=torch.long).view(n_blocks, BLOCK_SIZE)
    return blocks


def make_eval_set(held_out_texts, n_items=60, seed=123):
    rng = random.Random(seed)
    items = []
    for text in held_out_texts:
        lowered = text.lower()
        for term in ANCHOR_TERMS:
            idx = lowered.find(term)
            if idx == -1:
                continue
            prefix = text[max(0, idx - 300): idx]
            if len(prefix) < 40:
                continue
            decoys = rng.sample([t for t in ANCHOR_TERMS if t != term], k=3)
            items.append({"prefix": prefix, "correct": term, "decoys": decoys})
            break
    rng.shuffle(items)
    return items[:n_items]


def eval_domain_accuracy(model, tokenizer, eval_items):
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in eval_items:
            candidates = [item["correct"]] + item["decoys"]
            losses = []
            for cand in candidates:
                text = item["prefix"] + " " + cand
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=400).to(DEVICE)
                out = model(**enc, labels=enc["input_ids"])
                losses.append(out.loss.item())
            if int(np.argmin(losses)) == 0:
                correct += 1
    return correct / len(eval_items)


def train_one_run(base_model_path, blocks, seed, lr=2e-5, epochs=1, batch_size=8):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=TRAIN_DTYPE).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n = blocks.shape[0]
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            batch = blocks[perm[i:i + batch_size]].to(DEVICE)
            if batch.shape[0] == 0:
                continue
            out = model(input_ids=batch, labels=batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
    return model, tokenizer


def main():
    print(f"device: {DEVICE}")
    docs = load_all_docs()
    print(f"total documents: {len(docs)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rng = random.Random(42)
    rng.shuffle(docs)
    n_holdout = max(1, int(len(docs) * EVAL_HOLD_OUT_FRAC))
    holdout_docs = docs[:n_holdout]
    train_pool_docs = docs[n_holdout:]
    print(f"held out {len(holdout_docs)} documents for domain accuracy eval; "
          f"{len(train_pool_docs)} available for continued pretraining")

    classifier_path = DATA / "classifier_sklearn.pkl"
    keyword_filtered, two_stage_filtered = build_filters(train_pool_docs, classifier_path)
    print(f"keyword-gate-alone filter: {len(keyword_filtered)} docs")
    print(f"two-stage (keyword + classifier) filter: {len(two_stage_filtered)} docs")

    kw_blocks_probe = tokenize_and_pack(tokenizer, keyword_filtered, max_tokens=10 ** 9)
    two_blocks_probe = tokenize_and_pack(tokenizer, two_stage_filtered, max_tokens=10 ** 9)
    kw_tokens_available = kw_blocks_probe.numel()
    two_tokens_available = two_blocks_probe.numel()
    token_budget = min(kw_tokens_available, two_tokens_available, MAX_TOKEN_BUDGET)
    print(f"tokens available: keyword-alone={kw_tokens_available}, two-stage={two_tokens_available}")
    print(f"equal token budget used for both arms: {token_budget} "
          f"(capped at {MAX_TOKEN_BUDGET} for wall-clock time; see Honest framing)")

    kw_blocks = kw_blocks_probe[: token_budget // BLOCK_SIZE]
    two_blocks = two_blocks_probe[: token_budget // BLOCK_SIZE]

    eval_items = make_eval_set(holdout_docs, n_items=60)
    print(f"domain-accuracy eval set: {len(eval_items)} items")

    import time
    results = {"token_budget": token_budget, "n_eval_items": len(eval_items), "runs": []}
    for seed in SEEDS:
        t0 = time.monotonic()
        model_kw, tok = train_one_run(MODEL_NAME, kw_blocks, seed)
        print(f"seed={seed} keyword-alone training done in {time.monotonic() - t0:.0f}s", flush=True)
        acc_kw = eval_domain_accuracy(model_kw, tok, eval_items)
        del model_kw
        torch.cuda.empty_cache()

        t1 = time.monotonic()
        model_two, tok = train_one_run(MODEL_NAME, two_blocks, seed)
        print(f"seed={seed} two-stage training done in {time.monotonic() - t1:.0f}s", flush=True)
        acc_two = eval_domain_accuracy(model_two, tok, eval_items)
        del model_two
        torch.cuda.empty_cache()

        print(f"seed={seed}: keyword-alone acc={acc_kw:.4f}  two-stage acc={acc_two:.4f}  "
              f"delta={100 * (acc_two - acc_kw):.1f} points")
        results["runs"].append({"seed": seed, "acc_keyword_alone": acc_kw, "acc_two_stage": acc_two,
                                 "delta_points": 100 * (acc_two - acc_kw)})

    deltas = [r["delta_points"] for r in results["runs"]]
    results["mean_delta_points"] = float(np.mean(deltas))
    results["seed_spread_points"] = float(np.max(deltas) - np.min(deltas))
    results["outside_seed_spread"] = bool(min(deltas) > 0 or max(deltas) < 0)
    print(f"\nmean delta: {results['mean_delta_points']:.1f} points, "
          f"3-seed spread: {results['seed_spread_points']:.1f} points, "
          f"all-same-sign (outside spread): {results['outside_seed_spread']}")

    (DATA / "pretrain_ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", DATA / "pretrain_ablation_results.json")


if __name__ == "__main__":
    main()
