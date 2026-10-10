"""
Do retrieval confidence signals separate answerable from unanswerable questions?

First run gave cosine AUROC 0.660 against cross-encoder 0.828, on 22 negatives
that were almost all out-of-corpus: the governing law simply was not indexed.
That is the easy case. This version splits the negatives by type, because the
two failure modes are not the same problem:

  out_of_corpus        the answer's law is not indexed at all
  in_corpus_uncovered  the law IS indexed but does not address the question.
                       Topically adjacent paragraphs sit right there, so the
                       retriever returns high-scoring candidates and the answer
                       looks well-grounded. This is where confident fabrication
                       actually happens, and where an abstention signal has to
                       work if it is worth anything.

Prediction to record before running: separability drops on in_corpus_uncovered
for every signal, and drops furthest for cosine, which cannot tell "this passage
is about the right topic" from "this passage answers the question".

    pip install scikit-learn matplotlib

Usage:
    python eval/abstention_probe.py
    python eval/abstention_probe.py --rerank
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from qdrant_client import QdrantClient
from sklearn.metrics import roc_auc_score

from ingest import COLLECTION, embed_query, load_model
from run_eval import QUESTIONS, load_questions

OUT_DIR = Path(__file__).parent / "results"
K = 5

_ce = None


def load_ce():
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _ce


def signals(hits, question, use_rerank):
    """Candidate abstention signals, all from one retrieval call.

    top1      raw confidence in the best candidate
    gap       top1 minus top2. A large gap means one clear match; a small gap
              means everything looked equally plausible
    mean_top5 whether the whole neighbourhood is relevant or just one outlier
    rerank1   a learned relevance judgment rather than geometric proximity
    """
    scores = [h.score for h in hits]
    row = {
        "top1": scores[0] if scores else 0.0,
        "top2": scores[1] if len(scores) > 1 else 0.0,
        "mean_top5": sum(scores) / len(scores) if scores else 0.0,
    }
    row["gap"] = row["top1"] - row["top2"]

    if use_rerank:
        pairs = [(question, h.payload["text"]) for h in hits]
        row["rerank1"] = float(max(load_ce().predict(pairs))) if pairs else 0.0

    return row


def collect(use_rerank):
    rows = load_questions(QUESTIONS)
    model = load_model()
    qdrant = QdrantClient(url="http://localhost:6333")

    out = []
    for r in rows:
        hits = qdrant.query_points(
            collection_name=COLLECTION,
            query=embed_query(model, r["question"]).tolist(),
            limit=K,
        ).points
        answerable = bool(r["expected"])
        out.append({
            "question": r["question"],
            "lang": r.get("lang", "?"),
            "population": "in_scope" if answerable else "out_of_scope",
            # blank for answerable questions; the stratum for the rest
            "negative_type": "" if answerable else r.get("negative_type", "untagged"),
            "top_citation": hits[0].payload["citation"] if hits else "",
            **signals(hits, r["question"], use_rerank),
        })
    return out


def auroc(rows, signal):
    """1.0 = perfect separation, 0.5 = the signal carries no information."""
    y = [1 if r["population"] == "in_scope" else 0 for r in rows]
    x = [r[signal] for r in rows]
    if len(set(y)) < 2:
        return None
    return roc_auc_score(y, x)


def table(rows, signal_names, title):
    n_in = sum(1 for r in rows if r["population"] == "in_scope")
    n_out = len(rows) - n_in
    print(f"\n{title}   (answerable {n_in}, unanswerable {n_out})")
    print(f"{'signal':<12} {'AUROC':>7}   {'answerable':>11} {'unanswerable':>13}")
    print("-" * 48)
    for s in signal_names:
        a = auroc(rows, s)
        if a is None:
            continue
        m_in = sum(r[s] for r in rows if r["population"] == "in_scope") / max(n_in, 1)
        m_out = sum(r[s] for r in rows if r["population"] == "out_of_scope") / max(n_out, 1)
        print(f"{s:<12} {a:>7.3f}   {m_in:>11.3f} {m_out:>13.3f}")


def report(rows, signal_names):
    table(rows, signal_names, "ALL NEGATIVES")

    # The split that matters. Each stratum is scored against the same answerable
    # pool, so the AUROCs are directly comparable.
    in_scope = [r for r in rows if r["population"] == "in_scope"]
    strata = sorted({r["negative_type"] for r in rows if r["negative_type"]})

    for stratum in strata:
        subset = in_scope + [r for r in rows if r["negative_type"] == stratum]
        n = len(subset) - len(in_scope)
        if n < 8:
            print(f"\n{stratum}: only {n} negatives, skipping")
            continue
        table(subset, signal_names, f"NEGATIVES: {stratum}")

    print("\nper language (top1 AUROC, all negatives):")
    for lang in sorted({r["lang"] for r in rows}):
        subset = [r for r in rows if r["lang"] == lang]
        a = auroc(subset, "top1")
        print(f"  {lang:<4} n={len(subset):<4} " + (f"AUROC={a:.3f}" if a else "(single class)"))


def plot(rows, signal_names):
    OUT_DIR.mkdir(exist_ok=True)
    strata = sorted({r["negative_type"] for r in rows if r["negative_type"]})
    in_scope = [r for r in rows if r["population"] == "in_scope"]

    fig, axes = plt.subplots(len(strata), len(signal_names),
                             figsize=(5 * len(signal_names), 4 * len(strata)),
                             squeeze=False)

    for i, stratum in enumerate(strata):
        negs = [r for r in rows if r["negative_type"] == stratum]
        subset = in_scope + negs
        for j, s in enumerate(signal_names):
            ax = axes[i][j]
            ax.hist([r[s] for r in in_scope], bins=20, alpha=0.6,
                    label="answerable", color="tab:blue")
            ax.hist([r[s] for r in negs], bins=20, alpha=0.6,
                    label=stratum, color="tab:red")
            a = auroc(subset, s)
            ax.set_title(f"{s} vs {stratum}" + (f"  (AUROC {a:.3f})" if a else ""))
            ax.set_xlabel(s)
            ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "abstention_by_stratum.png"
    fig.savefig(path, dpi=120)
    print(f"\nplot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    rows = collect(args.rerank)
    names = ["top1", "gap", "mean_top5"] + (["rerank1"] if args.rerank else [])

    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "abstention_signals.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"csv -> {csv_path}")

    report(rows, names)
    plot(rows, names)


if __name__ == "__main__":
    main()
