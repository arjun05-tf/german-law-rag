"""
Measure retrieval quality against labelled questions.

The point of this file: every later change to the pipeline - hybrid search, a
reranker, a fine-tuned encoder - is a claim that retrieval got better. Without a
baseline that claim is unfalsifiable. This produces the baseline.

Metrics:
  recall@k  - was the correct paragraph anywhere in the top k
  MRR       - 1/rank of the correct paragraph, averaged. Punishes "right answer,
              wrong position", which recall@5 hides entirely and which is exactly
              the failure the German/English comparison surfaced.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --k 10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from ingest import COLLECTION, embed_query, load_model

QUESTIONS = Path(__file__).parent / "questions.jsonl"


def load_questions(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    missing = [r for r in rows if "question" not in r or "expected" not in r]
    if missing:
        raise ValueError(f"{len(missing)} rows missing 'question' or 'expected'")
    return rows


# def rank_of_expected(hits, expected):
#     """1-based position of the expected citation, or None if absent.

#     Compares on the citation string, which parse_law.py built. That is why the
#     citation format had to be exactly right at parse time - it is the join key
#     between the index and the ground truth.
#     """
#     for i, h in enumerate(hits, start=1):
#         if h.payload["citation"] == expected:
#             return i
#     return None

def rank_of_expected(hits, expected):
    """1-based rank of the first acceptable citation, or None.

    `expected` is a list: some questions are answered by more than one
    paragraph, and any of them counts as a correct retrieval.
    """
    wanted = set(expected)
    for i, h in enumerate(hits, start=1):
        if h.payload["citation"] in wanted:
            return i
    return None

def evaluate(k=5):
    rows = load_questions(QUESTIONS)
    rows = [r for r in rows if r["expected"]]
    model = load_model()
    qdrant = QdrantClient(url="http://localhost:6333")

    results = []
    for row in rows:
        hits = qdrant.query_points(
            collection_name=COLLECTION,
            query=embed_query(model, row["question"]).tolist(),
            limit=k,
        ).points
        results.append({**row, "rank": rank_of_expected(hits, row["expected"])})

    report(results, k)
    return results


def summarise(subset, k):
    """recall@1, recall@k and MRR for a group of results."""
    n = len(subset)
    if n == 0:
        return None
    ranks = [r["rank"] for r in subset]
    return {
        "n": n,
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        f"recall@{k}": sum(1 for r in ranks if r is not None) / n,
        "mrr": sum(1 / r for r in ranks if r is not None) / n,
    }


def report(results, k):
    overall = summarise(results, k)
    print(f"\n{'='*58}")
    print(f"n={overall['n']}  recall@1={overall['recall@1']:.2f}  "
          f"recall@{k}={overall[f'recall@{k}']:.2f}  MRR={overall['mrr']:.3f}")
    print("=" * 58)

    # Split by any label present in the questions file. Right now that is
    # `lang`, which turns the one-off German/English observation into a number
    # measured over the whole set.
    groups = defaultdict(list)
    for r in results:
        if "lang" in r:
            groups[r["lang"]].append(r)

    if groups:
        print("\nby language:")
        for name, subset in sorted(groups.items()):
            s = summarise(subset, k)
            print(f"  {name:<4} n={s['n']:<3} recall@1={s['recall@1']:.2f}  "
                  f"recall@{k}={s[f'recall@{k}']:.2f}  MRR={s['mrr']:.3f}")

    misses = [r for r in results if r["rank"] is None]
    demoted = [r for r in results if r["rank"] is not None and r["rank"] > 1]

    if misses:
        print(f"\nnot retrieved at all ({len(misses)}):")
        for r in misses:
            print(f"  [{r['expected']}] {r['question']}")

    if demoted:
        print(f"\nretrieved but not first ({len(demoted)}):")
        for r in sorted(demoted, key=lambda r: -r["rank"]):
            print(f"  rank {r['rank']}  [{r['expected']}] {r['question']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    evaluate(args.k)


if __name__ == "__main__":
    main()
