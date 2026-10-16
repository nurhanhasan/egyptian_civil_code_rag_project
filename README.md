# Egyptian Civil Code RAG

- [Egyptian Civil Code RAG](#egyptian-civil-code-rag)
  - [Run Q\&A in 3 commands](#run-qa-in-3-commands)
  - [Endpoints](#endpoints)
  - [Project layout](#project-layout)
  - [Compose Profiles](#compose-profiles)
  - [Quality gates](#quality-gates)
    - [Experimental tracking](#experimental-tracking)
    - [Evaluation](#evaluation)
    - [Load testing](#load-testing)
  - [Documentation](#documentation)


A Retrieval-Augmented Generation (RAG) system that answers questions about the
Egyptian Civil Code (Arabic legal Q&A) with citations. Non-commercial educational
use. Hallucinations are legally unacceptable here, so abstention and citations are
safety controls.


## Run Q&A in 3 commands

**Before you start (one-time):** create the environment file and point the generator
at an OpenAI-compatible endpoint (see [Generator](#generator) below).

```bash
cp .env.example .env.dev
```

**Then the three commands:**

```bash
# 1. Start the stack (API + Qdrant + Redis; no GPU required)
docker-compose down && docker-compose up -d --build

# 2. Build the index from the tracked corpus
docker-compose exec rag-api python -m ragapp.scripts.ingest_corpus

# 3. Ask a question
curl -X 'POST' \
  'http://localhost:8000/api/v1/ask' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "ما هو مصدر التشريع"
}'
```

The API is served on <http://localhost:8000>. Interactive docs: `/docs`.

## Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/api/v1/ask` | `{"question": str}` | `{"answer": str, "sources": [...]}` or an abstention `{"abstained": true, "reason": str, "answer": null, "sources": []}` |
| `GET` | `/api/v1/health` | — | `{"status": "ok"}` |

An empty or missing `question` is rejected with `422` (Pydantic validation).


## Project layout

```text
src/ragapp/
  core/            config, logging, cache, rate limiting
  entrypoints/     FastAPI app
  middleware/      request-id, logging, CORS
  modules/
    rag/           domain (ports/entities), adapters, factories, service, router
    health/
tests/
  unit/ integration/ e2e/ load/
docs/              PDR, Architecture, ADRs, implementation plans
data/corpus/       source + processed corpus (DVC-tracked)
```


For a quick offline check without any model server, set in `.env.dev`:

```bash
GENERATOR_BACKEND=fake
``` -->

## Models and revisions

| Role | Model | Revision |
|---|---|---|
| Embedder | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| Reranker | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Generator | `Qwen/Qwen3-0.6B` | served via vLLM |

All three are configurable through the environment (`BGE_M3_MODEL_PATH`,
`BGE_RERANKER_MODEL_PATH`, `GENERATOR_MODEL`). Pinning an immutable revision per
model is required.

## Run without Docker (development)

```bash
uv sync

# Make sure to have redis served on port 6379 and qdrant on port 6333 up and running

uv run uvicorn ragapp.entrypoints.fastapi.main:app --reload
```

On macOS, force the embedder/reranker onto CPU to avoid a Metal (MPS) assertion
under concurrency:

```bash
BGE_DEVICE=cpu uv run uvicorn ragapp.entrypoints.fastapi.main:app --reload
```

## Compose Profiles

- **default** — `rag-api`, `qdrant`, `redis`. Starts anywhere; the 3-command path.

- **`--profile gpu`** — adds the private `vllm` server (`Qwen/Qwen2.5-3B-Instruct-AWQ`),
  used instead of running vLLM locally. Needs an NVIDIA container runtime; the port
  is never published, `rag-api` reaches it at `http://vllm:8000/v1`. Pin `VLLM_IMAGE`.

- **NOTE**: Apple Silicon cannot GPU-accelerate the container: the `--profile gpu` service is for NVIDIA hosts only. To use the Apple GPU, run vLLM on the host with the community

  ```bash
  docker compose --profile gpu up -d
  ```

- **`--profile cpu`** — adds a CPU-only `vllm` (no GPU needed), served as the alias
  `vllm` so `rag-api` uses the same URL. AWQ is unsupported on the ARM CPU image
  (AWQ is x86/s390x/Power only), so it runs the unquantized
  `Qwen/Qwen3-0.6B` 

  ```bash
  docker compose --profile cpu up -d
  ```


## Quality gates

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
# uv run pytest tests/unit -q
# uv run pytest tests/integration -q
```

### Experimental tracking

To track experiments:

```bash
docker-compose -f docker-compose_qdrant.yaml up
docker-compose up mlflow
```

To evaluate the Query Path:
```bash
uv run python -m ragapp.scripts.evaluate ask "ما هو مصدر التشريع"  --tracking-uri http://localhost:5000
```

To evaluate the Offline Ingestion Path:
```bash
uv run python -m ragapp.scripts.ingest_corpus
```


### Evaluation

- Runs every gold question through the RAG pipeline, captures the generated
answer plus the retrieved contexts, and scores the run in MLflow with RAGAS.

```bash
uv run python -m ragapp.scripts.ragas --gold data/gold/gold_set.json \
        --judge-model qwen3.8-27b-8bit --with-embeddings \
            --base-url http://localhost:8000/v1  --strict
```

### Load testing

```bash
uv run locust -f tests/load/locustfile.py \
  --headless -u 3 -r 2 -t 10s  \
  --host http://localhost:8000 \
  --html reports/smoke.html \
  --stop-timeout 90s
```

See report `reports/load/report.html` for performance.


## Documentation

- `docs/PRD.md` — external brief.
- `docs/Architecture.md` — milestone plans.
<!-- - `docs/adr/` — architecture decision records. -->

