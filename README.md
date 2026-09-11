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
| 4. Retrieve | `ingest.py --ask` | top-k chunks with citations |
| 5. Generate | `gpt-4o-mini` | grounded answer with § references |

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

Overkill at 80 chunks, NumPy would do. Used because it's what scales, and it
gives persistence and metadata filtering for free.

**Grounding over recall**

The retriever always returns k chunks, cosine similarity has no notion of
"nothing matches," so an off-topic question still gets five confident-looking
paragraphs back. Refusal therefore has to happen in the generation step: the
model is instructed to answer only from the provided text and to say so when
that text doesn't cover the question. For legal QA a confident wrong answer
with a plausible citation is worse than no answer.

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

**3. Add your OpenAI key**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Used only for answer generation. Retrieval (`--ask`) works without it.

**4. Get the law**

```bash
curl -O https://www.gesetze-im-internet.de/arbzg/xml.zip
unzip xml.zip
python parse_law.py BJNR117100994.xml -o arbzg.jsonl
```

**5. Index and query**

```bash
python ingest.py

# retrieval only — see which paragraphs match
python ingest.py --ask "Wie lange darf ich am Tag arbeiten?"

# full answer, grounded in the retrieved paragraphs
python ingest.py --answer "Wie lange sind meine Ruhepausen?"

# refuses when the retrieved paragraphs don't cover the question
python ingest.py --answer "Wie hoch ist der Mindestlohn?"
```

---

## Status

**Working**
- XML to 80 chunks with citation metadata
- Dense retrieval over Qdrant
- Answer generation grounded in retrieved paragraphs, with refusal when the retrieved text doesn't cover the question

**Next**
- Eval set with ground-truth § references
- Hybrid retrieval + reranker, benchmarked
- FastAPI service, Docker Compose, CI

---

## Notes

Not legal advice.

Source: [gesetze-im-internet.de](https://www.gesetze-im-internet.de) — German
federal law is public domain under § 5 UrhG.
