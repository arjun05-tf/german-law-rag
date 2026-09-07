# german-law-rag

Question answering over German employment law, with exact paragraph citations.

Ask *"how long is my break entitlement?"* and get an answer sourced to
`§ 4 ArbZG` — not a page number, not a summary.

Currently indexes the **Arbeitszeitgesetz** (working time act). The pipeline
works on any federal law from gesetze-im-internet.de.

---

## Pipeline

| Step | Tool | Output |
|---|---|---|
| 1. Parse | `parse_law.py` | 80 Absatz-level chunks with § metadata |
| 2. Embed | `multilingual-e5-base` | 768-dim vectors |
| 3. Index | Qdrant | searchable collection |
| 4. Query | `ingest.py --ask` | top-k chunks with citations |

---

## Design decisions

**Chunked by Absatz, not by token count**

German law is cited as `§ 3 Abs. 1`. Fixed-size chunks cut across those
boundaries, so a retrieved chunk can't be attributed to one subsection. Absatz
is the smallest citable unit, so it's the natural chunk.

**Multilingual embeddings, not German-only**

Queries arrive in English against German source text. Multilingual models place
a sentence and its translation close together in one vector space; a
German-only model would score higher on German-to-German but can't bridge the
two languages.

**Qdrant**

Overkill at 80 chunks — NumPy would do. Used because it's what scales, and it
gives persistence and metadata filtering for free.

---

## Setup

**1. Start Qdrant**

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**2. Install**

```bash
pip install -r requirements.txt
```

**3. Get the law**

```bash
curl -O https://www.gesetze-im-internet.de/arbzg/xml.zip
unzip xml.zip
python parse_law.py BJNR117100994.xml -o arbzg.jsonl
```

**4. Index and query**

```bash
python ingest.py
python ingest.py --ask "Wie lange darf ich am Tag arbeiten?"
```

---

## Status

**Working**
- XML to 80 chunks with citation metadata
- Dense retrieval over Qdrant

**Next**
- Answer generation with enforced citations
- Eval set with ground-truth § references
- Hybrid retrieval + reranker, benchmarked
- FastAPI service, Docker Compose, CI

---

## Notes

Not legal advice.

Source: [gesetze-im-internet.de](https://www.gesetze-im-internet.de) — German
federal law is public domain under § 5 UrhG.
