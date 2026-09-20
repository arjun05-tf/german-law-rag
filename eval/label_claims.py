"""
Interactive labelling for claim attribution ground truth.

The slow part of labelling is not judgment, it is cross-referencing: read a
claim, then go find what the cited paragraph actually says. This puts both on
screen together, optionally with English translations, and takes a keypress.

The judgment stays human. That is the whole point: the column being filled in is
the ground truth that the cross-encoder gets measured against, so it cannot come
from a model without making the evaluation circular.

Progress saves after every keypress, so quitting and resuming is safe.

    pip install openai python-dotenv

Usage:
    python eval/label_claims.py                 # German only
    python eval/label_claims.py --translate     # adds English, needs OPENAI_API_KEY
"""

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results" / "attribution_claims.csv"
CHUNKS = Path(__file__).parent.parent / "arbzg.jsonl"
CACHE = Path(__file__).parent / "results" / "translation_cache.json"

WIDTH = 78


def load_paragraphs():
    """citation -> full paragraph text, so the cited source can be shown in full."""
    out = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            out[c["citation"]] = c["text"]
    return out


class Translator:
    """Translation is a reading aid, not a judgment, so a model is fine here.

    Cached on disk: the same paragraph is cited by many claims and re-translating
    it each time is wasted money and wasted seconds.
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        self.client = None
        if enabled:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv()
            self.client = OpenAI()

    def __call__(self, text):
        if not self.enabled or not text:
            return None
        if text in self.cache:
            return self.cache[text]
        r = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": f"Translate to English. Return only the translation.\n\n{text}"}],
            temperature=0,
        )
        out = r.choices[0].message.content.strip()
        self.cache[text] = out
        CACHE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return out


def wrap(label, text, indent=8):
    body = textwrap.fill(text, width=WIDTH - indent,
                         subsequent_indent=" " * indent)
    return f"{label:<{indent}}{body}"


def save(rows, fieldnames):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translate", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    paragraphs = load_paragraphs()
    tr = Translator(args.translate)

    todo = [i for i, r in enumerate(rows) if not r["supported_truth"].strip()]
    print(f"\n{len(rows)} claims, {len(todo)} unlabelled\n")
    print("For each claim, one question: does THIS paragraph say THIS?")
    print("Not whether the claim is true about German law generally.\n")
    print("  1 = the paragraph supports it")
    print("  0 = it does not (wrong paragraph, overreach, or invented)")
    print("  s = skip   q = save and quit\n")
    input("Enter to start...")

    for n, i in enumerate(todo, 1):
        row = rows[i]
        para = paragraphs.get(row["cited"], "(no paragraph cited)")

        print("\n" + "=" * WIDTH)
        print(f"[{n}/{len(todo)}]  {row['population']}  {row['lang']}  "
              f"model score {row['score']}")
        print("=" * WIDTH)

        print("\n" + wrap("CLAIM", row["claim"]))
        if args.translate:
            t = tr(row["claim"])
            if t:
                print(wrap("", t))

        print("\n" + wrap("CITED", f"[{row['cited']}] {para}"))
        if args.translate and row["cited"]:
            t = tr(para)
            if t:
                print(wrap("", t))

        while True:
            key = input("\n1 / 0 / s / q > ").strip().lower()
            if key in {"1", "0"}:
                row["supported_truth"] = key
                save(rows, fieldnames)
                break
            if key == "s":
                break
            if key == "q":
                save(rows, fieldnames)
                done = sum(1 for r in rows if r["supported_truth"].strip())
                print(f"\nsaved. {done}/{len(rows)} labelled.")
                sys.exit(0)

    save(rows, fieldnames)
    done = sum(1 for r in rows if r["supported_truth"].strip())
    print(f"\ndone. {done}/{len(rows)} labelled -> {CSV_PATH}")
    print("Now: python eval/attribution_eval.py --score")


if __name__ == "__main__":
    main()
