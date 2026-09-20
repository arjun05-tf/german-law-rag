"""
Claim-level evidence attribution.

Question-level abstention decides one thing for a whole answer: respond or
refuse. That misses the failure that actually happens, which is a mostly-grounded
answer with one unsupported sentence in it. The question was answerable, the
retrieval was fine, and a binary gate passes the whole thing through.

So score each claim separately. The probe showed cross-encoder scores separate
answerable from unanswerable questions (0.594 vs 0.159 mean, AUROC 0.828) where
cosine similarity does not (0.846 vs 0.828). This applies the same signal at
claim granularity. Whether it survives the finer granularity is an open question
and the reason eval/attribution_eval.py exists.

    pip install openai python-dotenv sentence-transformers
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

RERANKER = "BAAI/bge-reranker-v2-m3"

# From the probe: in_scope mean 0.594, out_of_scope mean 0.159. A cut in the
# 0.3-0.4 band sits between the populations rather than inside either. Provisional
# until tuned on labelled claims - this is a starting point, not a result.
SUPPORT_THRESHOLD = 0.35

_ce = None


def load_ce():
    global _ce
    if _ce is None:
        _ce = CrossEncoder(RERANKER)
    return _ce


CLAIM_PROMPT = """Split the answer below into individual factual claims.

Rules:
- One claim per statement of fact. A sentence stating two facts becomes two claims.
- Each claim must stand alone: resolve pronouns and implicit subjects.
- Keep the original language. Do not translate.
- Drop hedges, greetings and meta-commentary. Keep only assertions about the law.
- Return a JSON array of strings and nothing else. No markdown fence.

Answer:
{answer}"""


def extract_claims(answer_text, client, model="gpt-4o-mini"):
    """Decompose an answer into standalone claims.

    Sentence splitting would be cheaper, but legal prose packs several
    independent assertions into one sentence and splits badly on abbreviations
    ("Abs. 1", "Nr. 2"). Asking the model for claims handles both.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": CLAIM_PROMPT.format(answer=answer_text)}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to sentence splitting rather than losing the answer. Worse
        # decomposition beats no attribution.
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]

    return [c for c in claims if isinstance(c, str) and len(c) > 10]


def attribute(claims, hits):
    """Score every claim against every retrieved paragraph.

    Cost is len(claims) x len(hits) cross-encoder passes, so this is the
    expensive step. The argmax paragraph becomes the claim's citation, which is
    what makes the output verifiable: each sentence points at the specific
    Absatz it came from, not at the answer's general sources.
    """
    ce = load_ce()
    out = []

    for claim in claims:
        pairs = [(claim, h.payload["text"]) for h in hits]
        scores = ce.predict(pairs) if pairs else []

        if len(scores) == 0:
            out.append({"claim": claim, "score": 0.0, "citation": None, "supported": False})
            continue

        best = int(max(range(len(scores)), key=lambda i: scores[i]))
        score = float(scores[best])
        out.append({
            "claim": claim,
            "score": score,
            "citation": hits[best].payload["citation"],
            "supported": score >= SUPPORT_THRESHOLD,
        })

    return out


def render(attributed):
    """Answer with per-claim provenance.

    An unsupported claim is not removed. Removing it hides that the model
    produced it, and the whole point is to surface that.
    """
    lines = []
    for a in attributed:
        if a["supported"]:
            lines.append(f"  [{a['citation']}] {a['claim']}   ({a['score']:.2f})")
        else:
            lines.append(f"  [UNSUPPORTED] {a['claim']}   ({a['score']:.2f})")

    n_unsupported = sum(1 for a in attributed if not a["supported"])
    header = f"{len(attributed)} claims, {n_unsupported} unsupported"
    return header + "\n" + "\n".join(lines)


def attributed_answer(question, hits, answer_text, client=None):
    """Full pipeline: answer -> claims -> per-claim evidence."""
    client = client or OpenAI()
    claims = extract_claims(answer_text, client)
    return attribute(claims, hits)
