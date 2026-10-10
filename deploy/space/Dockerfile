# Backend for german-law-rag on a Hugging Face Space (Docker SDK).
#
# The code is cloned from GitHub rather than copied in, so this Space stays two
# small files and never drifts into a second copy of the project. Rebuild the
# Space to pick up new commits, or pin REF to a tag.
#
# Both models and the Qdrant index are baked at build time. They are ~3.5 GB of
# weights and the index needs the embedding model to exist, so doing it here
# turns a multi-minute first request into a normal one.

FROM python:3.11-slim

ARG REPO=https://github.com/arjun05-tf/german-law-rag
ARG REF=main
ARG LAW_XML=BJNR117100994.xml

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; everything it writes has to belong to it.
RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/app

RUN git clone --depth 1 --branch ${REF} ${REPO} app
WORKDIR /home/app/app

RUN pip install --no-cache-dir --user -r requirements.txt

# The parsed corpus is gitignored (it is derived), so derive it here.
RUN curl -sSfO https://www.gesetze-im-internet.de/arbzg/xml.zip \
    && unzip -o xml.zip \
    && python parse_law.py ${LAW_XML} -o arbzg.jsonl \
    && rm xml.zip

# Downloads both models and writes qdrant_data/, so the first query pays for
# neither. Fails the build rather than the first request if something is wrong.
RUN python -c "import serve, rerank; serve.startup(); rerank.load_reranker()"

EXPOSE 7860
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "7860"]
