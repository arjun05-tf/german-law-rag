"""
Contract check for the HTTP layer.

Two things can silently break and neither shows up in the retrieval eval:

  1. /api/query and /api/query/stream drifting apart. They share run_query, so
     the `done` event must carry exactly what the POST returns - the web UI
     reads one and anything scripted against the API reads the other.
  2. The stage events arriving out of order, or a stage claiming a score the
     pipeline has not computed yet. The UI renders candidates as soon as the
     dense stage lands, so `retrieved` must have cosine but no ce_score.

Runs against a server that is already up; it does not start one, because
loading the models takes most of a minute.

    uvicorn serve:app --port 8000
    python test_serve.py
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
QUESTION = "Wie lange sind meine Ruhepausen?"

# generate=false: the answer and attribution passes cost ~30s each on CPU and
# neither is part of the contract being checked here.
PARAMS = {"question": QUESTION, "k_retrieve": 20, "k": 5, "generate": "false"}

DOC_KEYS = {"question", "query_xy", "candidates", "evidence",
            "answer", "claims", "timing", "config"}
CAND_KEYS = {"id", "citation", "paragraph", "paragraph_title", "abschnitt",
             "absatz", "text", "chars", "cosine", "dense_rank", "ce_score",
             "final_rank", "xy"}


def get_json(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=300) as r:
        return json.load(r)


def post_json(path, body):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def stream(path):
    """Yield (event, payload) from an SSE response."""
    with urllib.request.urlopen(f"{BASE}{path}", timeout=300) as r:
        event = None
        for raw in r:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                yield event, json.loads(line[6:])
                event = None


def main():
    try:
        meta = get_json("/api/meta")
    except urllib.error.URLError as exc:
        print(f"no server at {BASE} ({exc}) - start it with "
              f"`uvicorn serve:app --port 8000`")
        return 2

    n = meta["chunks"]
    assert n > 0, "empty index"
    assert meta["dim"] > 0
    print(f"meta ok: {n} chunks, {meta['paragraphs']} paragraphs, dim {meta['dim']}")

    corpus = get_json("/api/corpus")["chunks"]
    assert len(corpus) == n, f"corpus {len(corpus)} != meta {n}"
    assert all(len(c["xy"]) == 2 for c in corpus), "every chunk needs a projection"
    print(f"corpus ok: {len(corpus)} chunks, all projected")

    # --- streaming ---------------------------------------------------------
    qs = urllib.parse.urlencode(PARAMS)
    seen, done = [], None
    for event, payload in stream(f"/api/query/stream?{qs}"):
        seen.append(event)
        if event == "retrieved":
            cands = payload["candidates"]
            assert len(cands) == PARAMS["k_retrieve"], len(cands)
            assert set(cands[0]) == CAND_KEYS, set(cands[0]) ^ CAND_KEYS
            # the reranker has not run; the UI must not be handed a score for it
            assert all(c["ce_score"] is None for c in cands), "ce_score before rerank"
            assert all(c["final_rank"] is None for c in cands), "final_rank before rerank"
            assert [c["dense_rank"] for c in cands] == list(range(1, len(cands) + 1))
            assert all(0 <= c["cosine"] <= 1 for c in cands)
        elif event == "reranked":
            cands = payload["candidates"]
            assert all(c["ce_score"] is not None for c in cands), "missing ce_score"
            kept = [c for c in cands if c["final_rank"] is not None]
            assert len(kept) == PARAMS["k"], f"{len(kept)} survivors, want {PARAMS['k']}"
            # final_rank must agree with the cross-encoder ordering it came from
            by_ce = sorted(kept, key=lambda c: -c["ce_score"])
            assert [c["final_rank"] for c in by_ce] == list(range(1, len(kept) + 1)), \
                "final_rank disagrees with ce_score order"
            assert payload["evidence"] == [c["citation"] for c in by_ce]
        elif event == "done":
            done = payload

    order = [e for e in ("embedded", "retrieved", "reranked", "done") if e in seen]
    assert seen == order, f"stages out of order: {seen}"
    assert done is not None, "stream ended without a done event"
    print(f"stream ok: {' -> '.join(seen)}")

    # --- the two endpoints must not drift ----------------------------------
    posted = post_json("/api/query", {"question": QUESTION, "k_retrieve": 20,
                                      "k": 5, "generate": False})
    assert set(posted) == DOC_KEYS, set(posted) ^ DOC_KEYS
    assert set(posted) == set(done), "POST and stream `done` disagree on keys"
    assert set(posted["candidates"][0]) == set(done["candidates"][0])
    assert posted["evidence"] == done["evidence"], "same question, different evidence"
    print(f"contract ok: POST == stream done, evidence {posted['evidence']}")

    # --- empty question is rejected, not silently answered ------------------
    try:
        post_json("/api/query", {"question": "   "})
        raise AssertionError("empty question was accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400, exc.code
    print("empty question rejected with 400")

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
