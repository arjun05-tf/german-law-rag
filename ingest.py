"""
Embed the parsed law chunks and load them into Qdrant.

Model choice: intfloat/multilingual-e5-base. Not because it's the strongest
German model - a German-only model would score better on German-to-German
retrieval. But queries here arrive in English ("how many hours can I work?")
against German source text, so we need a shared vector space across both
languages. A monolingual model can't do that.

    pip install sentence-transformers qdrant-client

First run downloads ~1.1GB of model weights.

Usage:
    python ingest.py                      # embed + upload
    python ingest.py --ask "Pausen"       # sanity-check retrieval
"""

import argparse
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
DIM = 768                    # must match the model; Qdrant rejects mismatches
COLLECTION = "gesetze"


def load_model():
    return SentenceTransformer(MODEL)


def embed_passages(model, texts):
    # e5 was trained with these prefixes and expects them at inference. Drop
    # them and quality degrades quietly - no error, just worse results.
    return model.encode([f"passage: {t}" for t in texts], show_progress_bar=True)


def embed_query(model, text):
    return model.encode(f"query: {text}")


def ingest(path):
    chunks = [json.loads(line) for line in open(path, encoding="utf-8")]
    print(f"{len(chunks)} chunks")

    model = load_model()
    vectors = embed_passages(model, [c["text"] for c in chunks])

    client = QdrantClient(url="http://localhost:6333")
    # recreate on every run: ingest should be idempotent while we're still
    # changing the chunking. Once the pipeline settles this becomes an upsert.
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

    client.upsert(
        collection_name=COLLECTION,
        points=[
            # Qdrant needs an integer or UUID id, so the human-readable chunk id
            # lives in the payload instead. Payload is what comes back on search
            # - the citation matters more than the vector at that point.
            PointStruct(id=i, vector=vec.tolist(), payload=chunk)
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ],
    )
    print(f"uploaded to '{COLLECTION}'")


def build_context(hits):
    parts = []
    for h in hits:
        p = h.payload
        # Score is cosine similarity. Watch the spread, not the absolute value:
        # if the top 5 all sit within ~0.02 of each other, retrieval isn't
        # actually discriminating and the reranker will have to earn its place.
        # print(f"[{h.score:.3f}] {p['citation']} - {p['paragraph_title']}")
        # print(f"        {p['text'][:160]}...\n")
        parts.append(f"[{p['citation']}] {p['text']}")
    return "\n\n".join(parts)

def ask(question, k=5):
    model = load_model()
    client = QdrantClient(url="http://localhost:6333")

    hits = client.query_points(
        collection_name=COLLECTION,
        query=embed_query(model, question).tolist(),
        limit=k,
    ).points

    context = build_context(hits)
    print(context)

SYSTEM_PROMPT = """You answer questions about German employment law.

Rules:
- Answer ONLY from the provided paragraphs. Do not use outside knowledge.
- Cite the paragraph you used, e.g. § 4 ArbZG.
- If the provided paragraphs do not contain the answer, say so plainly.
"""

def answer(question, k=5):
    model = load_model()
    client = QdrantClient(url="http://localhost:6333")
    hits = client.query_points(
        collection_name=COLLECTION,
        query=embed_query(model, question).tolist(),
        limit=k,
    ).points

    context = build_context(hits)

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Paragraphs: \n\n{context}\n\nQuestion: {question}"}
        ],
    )

    print(response.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="arbzg.jsonl")
    ap.add_argument("--ask")
    ap.add_argument("--answer")
    args = ap.parse_args()

    if args.answer:
        answer(args.answer)
    elif args.ask:
        ask(args.ask)
    else:
        ingest(args.chunks)


if __name__ == "__main__":
    main()
