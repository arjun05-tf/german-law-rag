"""
Do retrieval confidence signals separate answerable from unanswerable questions?

This is a probe, not a system change. It runs both populations through the
existing retriever, records several candidate confidence signals per question,
and reports how well each one separates them.

Populations come from questions.jsonl:
  in_scope      expected is non-empty: the corpus contains the answer
  out_of_scope  expected is empty: it does not

AUROC is the metric for RQ1 because it is threshold-free. Picking a threshold
first and reporting accuracy hides whether the signal is any good: a signal that
cannot separate the populations will still look fine at a threshold that just
predicts the majority class.

    pip install scikit-learn matplotlib

Usage:
    python eval/abstention_probe.py
    python eval/abstention_probe.py --rerank      # adds the cross-encoder signal
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


def signals(hits, question, use_rerank):
    """Candidate abstention signals, all derived from one retrieval call.

    top1      raw confidence in the best candidate
    gap       top1 minus top2. A large gap means the retriever found one clear
              match; a small gap means everything looked equally plausible, which
              is what an unanswerable question should look like
    mean_top5 whether the whole neighbourhood is relevant or just one outlier
    rerank1   a cross-encoder relevance judgment rather than geometric proximity.
              Prediction: this separates better than cosine, because cosine was
              never optimised for "is this sufficient evidence"
    """
    scores = [h.score for h in hits]
    row = {
        "top1": scores[0] if scores else 0.0,
        "top2": scores[1] if len(scores) > 1 else 0.0,
        "mean_top5": sum(scores) / len(scores) if scores else 0.0,
    }
    row["gap"] = row["top1"] - row["top2"]

    if use_rerank:
        from sentence_transformers import CrossEncoder
        global _ce
        try:
            _ce
        except NameError:
            _ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
        pairs = [(question, h.payload["text"]) for h in hits]
        row["rerank1"] = float(max(_ce.predict(pairs))) if pairs else 0.0

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
        out.append({
            "question": r["question"],
            "lang": r.get("lang", "?"),
            # The label IS the population: a non-empty expected list means the
            # corpus can answer it. This is why auditing those labels matters -
            # a mislabelled question lands in the wrong distribution.
            "population": "in_scope" if r["expected"] else "out_of_scope",
            "top_citation": hits[0].payload["citation"] if hits else "",
            **signals(hits, r["question"], use_rerank),
        })
    return out


def auroc(rows, signal):
    """Separability of the two populations by one signal.

    1.0 = perfect separation, 0.5 = the signal carries no information.
    Labelled so that in_scope is the positive class: higher confidence should
    mean more answerable.
    """
    y = [1 if r["population"] == "in_scope" else 0 for r in rows]
    x = [r[signal] for r in rows]
    if len(set(y)) < 2:
        return None
    return roc_auc_score(y, x)


def report(rows, signal_names):
    n_in = sum(1 for r in rows if r["population"] == "in_scope")
    n_out = len(rows) - n_in
    print(f"\nin_scope n={n_in}   out_of_scope n={n_out}")

    if n_out < 30:
        print(f"WARNING: {n_out} negatives is too few for calibration work. "
              "Enough to see whether separation exists, not to calibrate a threshold.")

    print(f"\n{'signal':<12} {'AUROC':>7}   {'in_scope mean':>14} {'out mean':>10}")
    print("-" * 50)
    for s in signal_names:
        a = auroc(rows, s)
        m_in = sum(r[s] for r in rows if r["population"] == "in_scope") / max(n_in, 1)
        m_out = sum(r[s] for r in rows if r["population"] == "out_of_scope") / max(n_out, 1)
        print(f"{s:<12} {a:>7.3f}   {m_in:>14.3f} {m_out:>10.3f}")

    # Per-language split: if separability differs, abstention needs different
    # thresholds per language, which is RQ3.
    print(f"\nper language (top1 AUROC):")
    for lang in sorted({r["lang"] for r in rows}):
        subset = [r for r in rows if r["lang"] == lang]
        a = auroc(subset, "top1")
        n = len(subset)
        print(f"  {lang:<4} n={n:<4} AUROC={a:.3f}" if a else f"  {lang:<4} n={n:<4} (single class)")


def plot(rows, signal_names):
    OUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(signal_names), figsize=(5 * len(signal_names), 4))
    if len(signal_names) == 1:
        axes = [axes]

    for ax, s in zip(axes, signal_names):
        for pop, colour in [("in_scope", "tab:blue"), ("out_of_scope", "tab:red")]:
            vals = [r[s] for r in rows if r["population"] == pop]
            ax.hist(vals, bins=20, alpha=0.6, label=pop, color=colour)
        ax.set_title(f"{s} (AUROC {auroc(rows, s):.3f})")
        ax.set_xlabel(s)
        ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "abstention_signals.png"
    fig.savefig(path, dpi=120)
    print(f"\nplot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    rows = collect(args.rerank)

    names = ["top1", "gap", "mean_top5"]
    if args.rerank:
        names.append("rerank1")

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
