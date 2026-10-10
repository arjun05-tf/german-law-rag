"""
Fold the expanded negative set into questions.jsonl, and fix two mislabels.

Two existing negatives are wrong: ArbZG § 6 Abs. 5 does provide for night-work
compensation ("eine angemessene Zahl bezahlter freier Tage oder einen
angemessenen Zuschlag"). Scoring those as unanswerable counts a correct answer
as a false acceptance, which biases the abstention result against the system.

Also tags every surviving negative with negative_type, because out-of-corpus and
in-corpus-but-uncovered are different problems: the first should be easy to
detect, the second is where topically adjacent paragraphs sit right next to the
question and confident fabrication actually happens. Reporting them together
hides that.

Usage (from the repo root):
    python eval/merge_negatives.py --dry-run
    python eval/merge_negatives.py
"""

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.jsonl"
NEGATIVES = HERE / "negatives.jsonl"

# Mislabelled: answerable from § 6 Abs. 5 after all.
PROMOTE = {
    "Wie hoch muss mein Lohn sein, wenn ich nachts arbeite?": ["§ 6 Abs. 5 ArbZG"],
    "Do night workers receive extra pay?": ["§ 6 Abs. 5 ArbZG"],
}

# Existing negatives, classified. Anything not listed defaults to out_of_corpus.
IN_CORPUS_UNCOVERED = {
    "Wie viele Stunden darf ich pro Woche arbeiten?",
    "How many hours am I allowed to work per week?",
    "Wie viele Pausen muss ich pro Woche machen?",
    "What is the maximum number of breaks per week?",
    "Wie viele Überstunden darf ich im Monat machen?",
    "How many overtime hours can I work per month?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8") if l.strip()]
    new_negs = [json.loads(l) for l in open(NEGATIVES, encoding="utf-8") if l.strip()]

    existing = {r["question"] for r in rows}
    promoted, tagged, added = 0, 0, 0

    for r in rows:
        if r["question"] in PROMOTE:
            r["expected"] = PROMOTE[r["question"]]
            r.pop("negative_type", None)
            promoted += 1
        elif not r["expected"]:
            r["negative_type"] = (
                "in_corpus_uncovered"
                if r["question"] in IN_CORPUS_UNCOVERED
                else "out_of_corpus"
            )
            tagged += 1

    for n in new_negs:
        if n["question"] not in existing:
            rows.append(n)
            added += 1

    pos = sum(1 for r in rows if r["expected"])
    neg = len(rows) - pos
    by_type = {}
    for r in rows:
        if not r["expected"]:
            by_type[r.get("negative_type", "?")] = by_type.get(r.get("negative_type", "?"), 0) + 1

    print(f"promoted to answerable: {promoted}")
    print(f"tagged existing negatives: {tagged}")
    print(f"added new negatives: {added}")
    print(f"\ntotal {len(rows)}: {pos} answerable, {neg} unanswerable")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    shutil.copy(QUESTIONS, QUESTIONS.with_suffix(".jsonl.bak"))
    with open(QUESTIONS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwritten -> {QUESTIONS}  (backup: {QUESTIONS.name}.bak)")


if __name__ == "__main__":
    main()
