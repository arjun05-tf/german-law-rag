---
title: german-law-rag API
emoji: ⚖️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Retrieval, reranking and claim attribution over the ArbZG
---

Backend for [german-law-rag](https://github.com/arjun05-tf/german-law-rag):
dense retrieval over 80 Absätze of the Arbeitszeitgesetz, cross-encoder
reranking, and claim-level attribution.

The code is cloned from GitHub at build time — this Space is only the
`Dockerfile` and this file. Rebuild it to pick up new commits.

| Route | Returns |
|---|---|
| `GET /api/meta` | models, dimensions, index source, corpus counts |
| `GET /api/corpus` | every indexed chunk with its § metadata and 2-D projection |
| `POST /api/query` | candidates, scores, evidence, answer, claims |
| `GET /api/query/stream` | the same pipeline as SSE, one event per stage |

`/` serves the web interface too, so this Space works on its own.

**Settings**

- `OPENAI_API_KEY` — a Space secret. Without it retrieval and reranking still
  run; answer generation and claim attribution are reported as skipped.
- `CORS_ORIGINS` — comma-separated origins allowed to call the API. Defaults to
  `*`. Set it to the frontend's origin once that is deployed.

On the free CPU tier the reranker scores 20 candidates in roughly a minute, so
queries are slow by design, not broken. The stream exists for exactly that: the
dense candidates render as soon as the vector search returns.
