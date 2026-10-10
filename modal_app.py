"""
Deploy serve.py on Modal.

The models are the reason this is not on a static or serverless-function host:
multilingual-e5-base and bge-reranker-v2-m3 together are ~3.5 GB of weights, so
the backend needs a container with real memory and a cold start it can amortise.

Nothing here reimplements the API. serve.app is the same FastAPI application
`uvicorn serve:app` runs locally; this only describes the image around it.

    pip install modal && modal setup
    modal deploy modal_app.py
"""
import modal

app = modal.App("german-law-rag")

# Layer order is deliberate: expensive and stable first, cheap and volatile last,
# so editing the frontend does not re-download 3.5 GB of weights.
SOURCES = ["serve.py", "ingest.py", "rerank.py", "attribute.py", "parse_law.py"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    # The default torch wheel carries CUDA. There is no GPU here, and the CPU
    # wheel is several gigabytes smaller, which is several seconds of cold start.
    .pip_install("torch", extra_options="--index-url https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("requirements.txt")
    # Both models, by name rather than through the app, so this layer survives
    # every source edit. The ids are the ones ingest.py and rerank.py load.
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('intfloat/multilingual-e5-base'); "
        "CrossEncoder('BAAI/bge-reranker-v2-m3')\""
    )
    .add_local_file("arbzg.jsonl", "/app/arbzg.jsonl", copy=True)
)

for source in SOURCES:
    image = image.add_local_file(source, f"/app/{source}", copy=True)

image = (
    image.workdir("/app")
    # Writes qdrant_data/ into the image, so a cold container starts against a
    # built index instead of embedding 80 chunks on the first request. The empty
    # web/ is there because importing serve mounts it as a static directory; the
    # runtime mount below fills it.
    .run_commands("mkdir -p /app/web", "python -c 'import serve; serve.startup()'")
    # Mounted at runtime, not baked: the frontend changes often and nothing in
    # the build depends on it.
    .add_local_dir("web", "/app/web")
)


@app.function(
    image=image,
    memory=4096,
    cpu=4.0,
    timeout=900,
    # Stay warm for five minutes after a request: the reranker is slow enough
    # that paying the cold start again mid-session is worse than the idle cost.
    scaledown_window=300,
    secrets=[modal.Secret.from_name("german-law-rag-openai")],
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():
    import serve

    return serve.app
