"""
HTTP service over the existing retrieval pipeline.

Nothing here reimplements retrieval. It imports the same functions the CLI and
the eval harness use - embed_query, rerank, attributed_answer - so what the web
UI shows is the pipeline that was measured, not a second copy of it that can
drift from it.

One thing is added for display only: a 2D PCA projection of the passage
vectors. The index is 768-dim and cosine-ranked; the projection is a view of
those real vectors, not the space retrieval runs in. Ranking never uses it.

    pip install -r requirements.txt
    uvicorn serve:app --port 8000

Qdrant: uses http://localhost:6333 when the `gesetze` collection is there.
Otherwise it embeds arbzg.jsonl into a local on-disk Qdrant under qdrant_data/
on first start, so the UI runs without Docker.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.decomposition import PCA

from ingest import (
    COLLECTION,
    DIM,
    MODEL,
    SYSTEM_PROMPT,
    build_context,
    embed_passages,
    embed_query,
    load_model,
)
from rerank import RERANKER, rerank

ROOT = Path(__file__).parent
CHUNKS = ROOT / "arbzg.jsonl"
LOCAL_QDRANT = ROOT / "qdrant_data"
GEN_MODEL = "gpt-4o-mini"

app = FastAPI(title="german-law-rag")
state = {}

# The frontend can be served from somewhere else than the pipeline - the models
# need more memory than a static host gives, so web/ can sit on a CDN while this
# runs on a box with RAM. Same-origin deploys are unaffected by this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- startup ---------------------------------------------------------------

def connect_qdrant():
    """Remote Qdrant if it holds the collection, else a local on-disk one."""
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=2.0)
        names = [c.name for c in client.get_collections().collections]
        if COLLECTION in names:
            return client, "http://localhost:6333"
    except Exception:
        pass

    client = QdrantClient(path=str(LOCAL_QDRANT))
    names = [c.name for c in client.get_collections().collections]
    if COLLECTION not in names:
        _build_local(client)
    return client, f"local:{LOCAL_QDRANT.name}"


def _build_local(client):
    if not CHUNKS.exists():
        raise RuntimeError(f"{CHUNKS} missing - run parse_law.py first")
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    vectors = embed_passages(state["model"], [c["text"] for c in chunks])
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=i, vector=v.tolist(), payload=c)
                for i, (c, v) in enumerate(zip(chunks, vectors))],
    )


def load_corpus(client):
    """Every indexed point with its vector, plus the display projection.

    PCA is fitted on the passage vectors once. A query is projected with the
    same fitted transform so the two land in one picture - the only way the
    query marker means anything relative to the corpus cloud.
    """
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION, limit=256,
            with_vectors=True, with_payload=True, offset=offset,
        )
        points.extend(batch)
        if offset is None:
            break

    points.sort(key=lambda p: p.payload["id"])
    vectors = np.array([p.vector for p in points], dtype=np.float32)
    pca = PCA(n_components=2).fit(vectors)
    xy = pca.transform(vectors)
    # Scale to roughly [-1, 1] so the client never has to know the data range.
    scale = float(np.abs(xy).max()) or 1.0

    chunks = []
    for p, (x, y) in zip(points, xy / scale):
        chunks.append({**p.payload, "xy": [float(x), float(y)], "chars": len(p.payload["text"])})
    return {"chunks": chunks, "pca": pca, "scale": scale,
            "explained": [float(v) for v in pca.explained_variance_ratio_]}


@app.on_event("startup")
def startup():
    t0 = time.time()
    state["model"] = load_model()
    client, source = connect_qdrant()
    state["qdrant"] = client
    state["source"] = source
    state["corpus"] = load_corpus(client)
    state["boot_s"] = round(time.time() - t0, 1)


def project(vector):
    corpus = state["corpus"]
    xy = corpus["pca"].transform(np.array([vector], dtype=np.float32))[0] / corpus["scale"]
    return [float(xy[0]), float(xy[1])]


# --- api -------------------------------------------------------------------

class Query(BaseModel):
    question: str
    k_retrieve: int = 20
    k: int = 5
    generate: bool = True


@app.get("/api/meta")
def meta():
    c = state["corpus"]
    by_abschnitt = {}
    for chunk in c["chunks"]:
        by_abschnitt.setdefault(chunk["abschnitt"], []).append(chunk["citation"])
    return {
        "embed_model": MODEL,
        "dim": DIM,
        "reranker": RERANKER,
        "gen_model": GEN_MODEL,
        "generation_available": bool(os.getenv("OPENAI_API_KEY")),
        "index": state["source"],
        "collection": COLLECTION,
        "chunks": len(c["chunks"]),
        "paragraphs": len({x["paragraph"] for x in c["chunks"]}),
        "sections": list(by_abschnitt),
        "gesetz": c["chunks"][0]["gesetz"] if c["chunks"] else "",
        "pca_explained": c["explained"],
        "boot_s": state["boot_s"],
    }


@app.get("/api/corpus")
def corpus():
    """The indexed chunks - real § metadata, text and projection coordinates."""
    return {"chunks": state["corpus"]["chunks"]}


def run_query(question, k_retrieve, k, generate):
    """The pipeline, as a generator of (stage, payload) as each stage finishes.

    One implementation feeds both endpoints: /api/query drains it and returns
    the final payload, /api/query/stream forwards each stage as it lands. The
    stage events are what actually happened, measured here - the UI never has
    to guess where the request is, and it is never shown a stage that has not
    run. On CPU the reranker and the attribution pass dominate (tens of
    seconds), which is the whole reason the stream exists.
    """
    question = question.strip()
    if not question:
        raise HTTPException(400, "empty question")
    k_retrieve = max(1, min(k_retrieve, len(state["corpus"]["chunks"])))
    k = max(1, min(k, k_retrieve))
    timing = {}

    t = time.perf_counter()
    qvec = embed_query(state["model"], question)
    timing["embed_ms"] = round((time.perf_counter() - t) * 1000)
    query_xy = project(qvec)
    yield "embedded", {"query_xy": query_xy, "embed_ms": timing["embed_ms"]}

    t = time.perf_counter()
    hits = state["qdrant"].query_points(
        collection_name=COLLECTION, query=qvec.tolist(), limit=k_retrieve,
    ).points
    timing["search_ms"] = round((time.perf_counter() - t) * 1000)

    xy_by_id = {c["id"]: c["xy"] for c in state["corpus"]["chunks"]}
    candidates = []
    for i, h in enumerate(hits, start=1):
        p = h.payload
        candidates.append({
            "id": p["id"],
            "citation": p["citation"],
            "paragraph": p["paragraph"],
            "paragraph_title": p["paragraph_title"],
            "abschnitt": p["abschnitt"],
            "absatz": p["absatz"],
            "text": p["text"],
            "chars": len(p["text"]),
            "cosine": float(h.score),
            "dense_rank": i,
            "ce_score": None,
            "final_rank": None,
            "xy": xy_by_id.get(p["id"]),
        })
    # Dense stage only: ce_score and final_rank are still null here because the
    # reranker has not run. The client renders them as pending, not as zero.
    yield "retrieved", {"candidates": candidates, "search_ms": timing["search_ms"]}

    t = time.perf_counter()
    # with_scores keeps the demoted candidates' scores - one scoring pass, and
    # the UI can show what moved down as well as what moved up.
    reranked, ce = rerank(question, hits, top_k=k, with_scores=True)
    timing["rerank_ms"] = round((time.perf_counter() - t) * 1000)

    final_rank = {h.id: i + 1 for i, h in enumerate(reranked)}
    for c, h in zip(candidates, hits):
        c["ce_score"] = ce.get(h.id)
        c["final_rank"] = final_rank.get(h.id)

    evidence = [c["citation"] for c in sorted(
        (c for c in candidates if c["final_rank"]), key=lambda c: c["final_rank"])]
    yield "reranked", {"candidates": candidates, "evidence": evidence,
                       "rerank_ms": timing["rerank_ms"]}

    answer_text, claims = None, None
    if generate and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        from attribute import attributed_answer

        try:
            client = OpenAI()
            t = time.perf_counter()
            response = client.chat.completions.create(
                model=GEN_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Paragraphs: \n\n{build_context(reranked)}\n\nQuestion: {question}"},
                ],
            )
            answer_text = response.choices[0].message.content
            timing["generate_ms"] = round((time.perf_counter() - t) * 1000)
            yield "answered", {"answer": answer_text, "generate_ms": timing["generate_ms"]}

            t = time.perf_counter()
            claims = attributed_answer(question, reranked, answer_text, client)
            timing["attribute_ms"] = round((time.perf_counter() - t) * 1000)
            yield "attributed", {"claims": claims, "attribute_ms": timing["attribute_ms"]}
        except Exception as exc:  # the answer is optional; retrieval is not
            answer_text, claims = None, None
            timing["generate_error"] = str(exc)[:200]
            yield "generate_error", {"error": timing["generate_error"]}

    yield "done", {
        "question": question,
        "query_xy": query_xy,
        "candidates": candidates,
        "evidence": evidence,
        "answer": answer_text,
        "claims": claims,
        "timing": timing,
        "config": {"k_retrieve": k_retrieve, "k": k},
    }


@app.post("/api/query")
def query(q: Query):
    for stage, payload in run_query(q.question, q.k_retrieve, q.k, q.generate):
        if stage == "done":
            return payload


@app.get("/api/query/stream")
def query_stream(question: str, k_retrieve: int = 20, k: int = 5, generate: bool = True):
    """Same pipeline, one SSE event per completed stage.

    GET rather than POST so EventSource can be used directly - no streaming
    body, no fetch-reader plumbing on the client for what is a handful of
    events.
    """
    def events():
        try:
            for stage, payload in run_query(question, k_retrieve, k, generate):
                yield f"event: {stage}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except HTTPException as exc:
            yield f"event: error\ndata: {json.dumps({'error': exc.detail})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)[:200]})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --- static ----------------------------------------------------------------

WEB = ROOT / "web"


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
