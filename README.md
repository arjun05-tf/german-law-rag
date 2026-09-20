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

## Results

Retrieval measured on 91 hand-labelled questions with ground-truth § references,
in German and English.

<!-- | Config | recall@1 | recall@5 | MRR |
|---|---|---|---|
| Dense (multilingual-e5-base) | 0.44 | 0.84 | 0.585 |
| + cross-encoder reranker | | | |

**By query language**

| | n | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| German | 46 | 0.59 | 0.83 | 0.670 |
| English | 45 | 0.29 | 0.84 | 0.498 |

Recall@5 is near-identical across languages, but recall@1 is twice as high in
German. The correct paragraph is found equally often and ranked worse, a
ranking problem, not a retrieval one, which is what motivates the reranker.

16% of questions miss entirely at k=5. These are mostly scenario questions
("I worked until midnight and start at 6 a.m., is that allowed?") where the
user describes a situation and the law states a rule. No reranker fixes those;
the paragraph never enters the candidate set. -->

| Config | recall@1 | recall@5 | MRR |
|---|---|---|---|
| Dense (top-5) | 0.44 | 0.84 | 0.585 |
| Dense top-20 + cross-encoder rerank | 0.60 | 0.90 | 0.722 |

**By query language**

| | recall@1 (dense) | recall@1 (reranked) | recall@5 (dense) | recall@5 (reranked) |
|---|---|---|---|---|
| German | 0.59 | 0.59 | 0.83 | 0.89 |
| English | 0.29 | 0.60 | 0.84 | 0.91 |

Reranking doubled English recall@1 and left German unchanged, closing a 30-point
cross-lingual gap to one point. A cross-encoder can only reorder candidates, so
this is direct evidence that the English deficit was a ranking failure rather than
a retrieval failure — the correct paragraph was already being found and ordered
badly.

Caveat: recall@5 also rose in both languages. Reordering five results cannot change
whether the answer is among them, so that gain comes from widening the candidate
pool from 5 to 20 before reranking, not from the reranker itself. The two effects
are confounded in this run and would need separating to attribute cleanly.

---

### Abstention: can retrieval confidence tell answerable from unanswerable?

The retriever always returns k chunks, so an off-topic question still gets five
confident-looking paragraphs back. That raises the question of whether any
retrieval signal indicates when the corpus cannot answer at all.

Measured on 91 answerable and 22 unanswerable questions. AUROC is threshold-free:
1.0 is perfect separation, 0.5 is no information.

| Signal | AUROC | answerable (mean) | unanswerable (mean) |
|---|---|---|---|
| Cosine top-1 | 0.660 | 0.846 | 0.828 |
| Top-1 minus top-2 gap | 0.682 | 0.007 | 0.003 |
| Mean top-5 cosine | 0.644 | 0.836 | 0.821 |
| **Cross-encoder top-1** | **0.828** | **0.594** | **0.159** |

Cosine similarity separates the two populations by 0.018. The cross-encoder
separates them by 0.435, roughly twenty-four times wider. Embedding similarity
measures topical proximity, not whether a passage constitutes sufficient
evidence, so a threshold on cosine score cannot support abstention here. A
relevance model recovers most of that signal.

Cosine separability also differs by language (German AUROC 0.806, English 0.669),
a second cross-lingual asymmetry independent of the ranking gap above.

Caveat: 22 negatives is enough to show separation exists, not enough to calibrate
a threshold. They are also almost all out-of-corpus questions; the harder case is
in-corpus-but-uncovered, where topically adjacent paragraphs exist.

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
- Absatz-level parsing of official Bundesrecht XML into 80 chunks with § metadata
- Dense retrieval over Qdrant with multilingual embeddings
- Cross-encoder reranking, benchmarked against the dense baseline
- Retrieval evaluation on 91 hand-labelled questions (recall@1, recall@5, MRR,
  split by query language)
- Abstention analysis: four retrieval confidence signals compared by AUROC on
  answerable vs unanswerable questions
- Answer generation grounded in retrieved paragraphs, with refusal when the
  retrieved text doesn't cover the question

**In progress**
- Claim-level evidence attribution: answers are decomposed into individual claims
  and each is scored against the retrieved paragraphs, so support is reported per
  sentence rather than per answer. Implemented; awaiting hand-labelled ground
  truth to measure whether the cross-encoder signal holds at claim granularity
  (0.828 at question level)

**Next**
- Claim taxonomy at extraction: meta-statements about the corpus are not
  evidenceable and currently pollute attribution
- Expand the negative set to ~60, stratified into out-of-corpus and
  in-corpus-but-uncovered
- Separate the reranking effect from the wider candidate pool (rerank top-5 only)
- FastAPI service with the model loaded once at startup, Docker Compose, CI that
  fails on retrieval regression
- Additional laws (BUrlG, MiLoG) with metadata filtering and a router

---

## Notes

Not legal advice.

Source: [gesetze-im-internet.de](https://www.gesetze-im-internet.de) — German
federal law is public domain under § 5 UrhG.
