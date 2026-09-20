"""
Does the cross-encoder signal survive at claim granularity?

The probe validated it at question level: AUROC 0.828 separating answerable from
unanswerable questions. Claim-level attribution assumes the same signal works on
a single sentence against a single paragraph. That is an assumption, not a
result, and this measures it.

Procedure:
  1. --generate   run questions through the pipeline, write claims with scores
                  and a blank `supported_truth` column
  2. label        open the CSV, mark each claim 1 (supported by the cited
                  paragraph) or 0 (not). This is the slow part and cannot be
                  automated: the whole point is a human judgment to measure
                  the model against
  3. --score      AUROC of the cross-encoder score against your labels

Deliberately includes out-of-scope questions. Those are where fabricated claims
should appear, and an attribution system that only sees well-grounded answers
has not been tested.

Usage:
    python eval/attribution_eval.py --generate --n 20
    python eval/attribution_eval.py --score
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from qdrant_client import QdrantClient
from sklearn.metrics import roc_auc_score

from attribute import SUPPORT_THRESHOLD, attribute, extract_claims
from ingest import COLLECTION, SYSTEM_PROMPT, build_context, embed_query, load_model
from run_eval import QUESTIONS, load_questions

OUT = Path(__file__).parent / "results" / "attribution_claims.csv"
K = 5


def generate(n):
    rows = load_questions(QUESTIONS)

    # Mix both populations. Out-of-scope questions are where unsupported claims
    # come from, so an evaluation without them measures the easy case only.
    in_scope = [r for r in rows if r["expected"]][: int(n * 0.6)]
    out_scope = [r for r in rows if not r["expected"]][: n - len(in_scope)]
    sample = in_scope + out_scope

    model = load_model()
    qdrant = QdrantClient(url="http://localhost:6333")
    client = OpenAI()

    records = []
    for i, row in enumerate(sample, 1):
        hits = qdrant.query_points(
            collection_name=COLLECTION,
            query=embed_query(model, row["question"]).tolist(),
            limit=K,
        ).points

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Paragraphs:\n\n{build_context(hits)}\n\nQuestion: {row['question']}"},
            ],
        )
        answer = response.choices[0].message.content

        claims = extract_claims(answer, client)
        for a in attribute(claims, hits):
            records.append({
                "question": row["question"],
                "lang": row.get("lang", "?"),
                "population": "in_scope" if row["expected"] else "out_of_scope",
                "claim": a["claim"],
                "cited": a["citation"] or "",
                "score": round(a["score"], 4),
                "predicted_supported": int(a["supported"]),
                "supported_truth": "",   # you fill this in
            })

        print(f"  {i}/{len(sample)}  {len(claims)} claims  {row['question'][:55]}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print(f"\n{len(records)} claims -> {OUT}")
    print("Label the supported_truth column (1/0), then run with --score.")


def score():
    rows = [r for r in csv.DictReader(open(OUT, encoding="utf-8"))]
    labelled = [r for r in rows if r["supported_truth"].strip() in {"0", "1"}]

    if len(labelled) < 20:
        print(f"Only {len(labelled)} of {len(rows)} labelled. Label more first.")
        return

    y = [int(r["supported_truth"]) for r in labelled]
    x = [float(r["score"]) for r in labelled]

    print(f"\nlabelled claims: {len(labelled)}")
    print(f"actually supported: {sum(y)} ({sum(y)/len(y):.0%})")

    if len(set(y)) < 2:
        print("All labels identical - AUROC undefined.")
        return

    print(f"\nclaim-level AUROC: {roc_auc_score(y, x):.3f}")
    print("question-level AUROC was 0.828 - compare directly.")

    # Confusion at the current threshold. AUROC says whether the signal ranks
    # correctly; this says what the chosen cut actually does, and the two
    # error types have very different costs here.
    tp = sum(1 for r in labelled if int(r["predicted_supported"]) and int(r["supported_truth"]))
    fp = sum(1 for r in labelled if int(r["predicted_supported"]) and not int(r["supported_truth"]))
    fn = sum(1 for r in labelled if not int(r["predicted_supported"]) and int(r["supported_truth"]))
    tn = sum(1 for r in labelled if not int(r["predicted_supported"]) and not int(r["supported_truth"]))

    print(f"\nat threshold {SUPPORT_THRESHOLD}:")
    print(f"  passed an unsupported claim (false accept): {fp}")
    print(f"  flagged a supported claim (false reject):   {fn}")
    print(f"  correct: {tp + tn}/{len(labelled)}")

    for pop in ("in_scope", "out_of_scope"):
        subset = [r for r in labelled if r["population"] == pop]
        if len(subset) >= 10 and len({int(r["supported_truth"]) for r in subset}) == 2:
            a = roc_auc_score([int(r["supported_truth"]) for r in subset],
                              [float(r["score"]) for r in subset])
            print(f"  {pop:<14} n={len(subset):<4} AUROC={a:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    if args.generate:
        generate(args.n)
    elif args.score:
        score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
