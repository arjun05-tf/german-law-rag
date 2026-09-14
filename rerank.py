"""
Cross-encoder reranking.

Why this exists: the bi-encoder embeds query and passage separately and compares
vectors, so it never sees the pair together. Fast, but it produced recall@1 0.44
against recall@5 0.84 - the right paragraph is usually in the candidate set and
merely ordered badly. A cross-encoder reads (query, passage) jointly and scores
relevance directly. Far slower, so it only runs on a shortlist the bi-encoder
already narrowed.

It cannot rescue a paragraph the bi-encoder never retrieved. recall@N of the
first stage is therefore a hard ceiling on what this can achieve.

    pip install -U sentence-transformers
"""

from sentence_transformers import CrossEncoder

RERANKER = "BAAI/bge-reranker-v2-m3"

_model = None


def load_reranker():
    # Cached at module level: loading is seconds and the eval calls this 91 times.
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER)
    return _model


def rerank(question, hits, top_k=5):
    """Reorder hits by cross-encoder relevance, return the top_k.

    Scores are not comparable to the bi-encoder's cosine similarity - different
    model, different scale. Only the resulting order is meaningful.
    """
    if not hits:
        return hits

    model = load_reranker()
    pairs = [(question, h.payload["text"]) for h in hits]
    scores = model.predict(pairs)

    ranked = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
    return [h for h, _ in ranked[:top_k]]
