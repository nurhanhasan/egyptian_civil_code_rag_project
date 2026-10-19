# Egyptian Civil Code RAG

- [Egyptian Civil Code RAG](#egyptian-civil-code-rag)
  - [How it works](#how-it-works)
    - [Components](#components)
    - [Models and revisions](#models-and-revisions)
    - [Project layout](#project-layout)
  - [Quickstart](#quickstart)
    - [Run Q\&A in 3 commands](#run-qa-in-3-commands)
    - [Run without Docker (development)](#run-without-docker-development)
    - [Endpoints](#endpoints)
    - [Compose Profiles](#compose-profiles)
  - [Quality gates](#quality-gates)
    - [Experimental tracking](#experimental-tracking)
    - [Evaluation](#evaluation)
    - [Load testing](#load-testing)
  - [Canary Deployment](#canary-deployment)
  - [Documentation](#documentation)


A Retrieval-Augmented Generation (RAG) system that answers questions about the
Egyptian Civil Code (Arabic legal Q&A) with citations. Non-commercial educational
use. Hallucinations are legally unacceptable here, so abstention and citations are
safety controls.


## How it works

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3--ogs382Na1U2v3LfVVcQ.png)

### Components

| Component | Role |
|---|---|
| **`rag-api` (FastAPI)** | Owns `POST /api/v1/ask` and `GET /api/v1/health`; validation, rate limiting, error envelope, request/trace IDs, response shaping. |
| **`RAGPipelineService`** | Orchestrates embed → retrieve → rerank → context assembly → generate → citation/abstention. |
| **BGE-M3 embedder** | Dense + sparse vectors, pinned revision. BentoML-decorated adapter, currently loaded in-process by `rag-api`. |
| **BGE reranker** | `bge-reranker-v2-m3` cross-encoder. BentoML-decorated adapter, currently loaded in-process. |
| **Qdrant** | Hybrid index with named dense and sparse vectors, RRF fusion, payload indexes on `jurisdiction` and `doc_status`, immutable build collections and a stable alias. |
| **vLLM** | Qwen generator served over an OpenAI-compatible `/v1/chat/completions` API. Never imported as a library. CPU and GPU profiles. |
| **Redis** | Cache and distributed rate-limit counters. Celery broker/result DBs are reserved but no workers are enabled yet. |
| **MLflow** | Experiment tracking and evaluation runs (`--profile mlflow`). |
| **Langfuse** | Request traces and RAG chain observations. |

* **One public edge.** FastAPI owns `/api/v1/ask` end-to-end. BentoML does not
  expose the route, and vLLM is never the public API. In the Compose topology,
  `rag-api` (and `nginx` in the `compose` profile) is the application entrypoint;
  Qdrant, Redis, vLLM and the Langfuse stores live on the private `appnet` network.
  Local development still publishes dependency ports for convenience.

* **Immutable builds, alias promotion.** Ingestion creates a new collection named
  by its build ID, populates it, then moves the stable alias
  (`egyptian-civil-code`) to it. `/ask` reads only the alias. Promotion and
  rollback are alias changes — never an image rebuild or a mutation of a live
  collection.

* **Traceability.** The edge assigns `X-Request-ID`; the RAG pipeline returns
  `X-Trace-ID`. The trace ID groups the Langfuse observations
  (`embedding-question`, `retrieving`, `reranker`, `generate`, `citation`) and
  lets a response be correlated with its logs and trace.

### Models and revisions

| Role | Model | Revision |
|---|---|---|
| Embedder | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| Retriever | `Qdrant` | `"qdrant-client>=1.19.1"` |
| Reranker | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Generator | `Qwen/Qwen3-0.6B` | served via vLLM (on CPU inside docker) |

All are configurable through the environment (`BGE_M3_MODEL_PATH`
`BGE_RERANKER_MODEL_PATH`, `GENERATOR_MODEL`). Pinning an immutable revision per
model is required.

### Project layout

```text
src/ragapp/
  core/            config, logging, cache, rate limiting
  entrypoints/     FastAPI app
  middleware/      request-id, logging, CORS
  modules/
    rag/           domain (ports/entities), adapters, factories, schemas, service, router
    health/        service, router
  scripts/         evaluate.py, ingest_corpus.py, ragas.py
tests/
  unit/ integration/ e2e/ load/
docs/              PDR, Architecture, ADRs, implementation plans
data/corpus/       source + processed corpus (DVC-tracked)
```

* For the MLOps Maturity breakdown, visit the branches in this order
  1. [no-mlops](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/no-mlops)
  2. [devops-but-not-mlops](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/devops-but-not-mlops)
  3. [mlops-w-atomated-data-pipeline](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/mlops-w-atomated-data-pipeline)
  4. [mlops-automated-deployment](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/mlops-automated-deployment)


## Quickstart

### Run Q&A in 3 commands

  ```bash
  # 1. Clone
  git clone https://github.com/nurhanhasan/egyptian_civil_code_rag_project && cd egyptian_civil_code_rag_project && cp .env.example .env.dev # edit, if needed.

  # 2. Start CPU profile. For GPU, pick the `gpu` one.
  docker-compose --profile cpu --profile mlflow --profile compose up -d

  # 3. Ask a question
  curl -i -X 'POST' \
    'http://localhost/api/v1/ask' \
    -H 'accept: */*' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "متى تسقط الدعوى المدنية بالتقادم؟"
  }'


  # And, for abstention

  curl -i -X 'POST' \
    'http://localhost/api/v1/ask' \
    -H 'accept: */*' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "ما عقوبة جريمة القتل"             
  }'

  # finally,
  docker-compose --profile cpu --profile mlflow --profile compose down
  ```

* The API is served on <http://localhost:8000>. Interactive docs: [/docs](http://localhost:8000/docs).

* For artifacts and screenshots, please check [Here](./reports/README.md).

* To build the index
  ```bash
  docker-compose exec rag-api python -m ragapp.scripts.ingest_corpus
  ```

### Run without Docker (development)

  ```bash
  uv sync

  docker-compose --profile cpu --profile mlflow up                  # deps without the api
  uv run uvicorn ragapp.entrypoints.fastapi.main:app --reload       # OR the debugger config
  ```

On macOS, force the embedder/reranker onto CPU to avoid a Metal (MPS) assertion
under concurrency:

  ```bash
  BGE_DEVICE=cpu uv run uvicorn ragapp.entrypoints.fastapi.main:app --reload
  ```


### Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/api/v1/ask` | `{"question": str}` | `{"answer": str, "sources": [...]}` or an abstention `{"abstained": true, "reason": str, "answer": null, "sources": []}` |
| `GET` | `/api/v1/health` | — | `{"status":"healthy","documents_indexed":N}` |

An empty or missing `question` is rejected with `422` (Pydantic validation).



### Compose Profiles


| Profile   | Services    |
|-----------|-------------|
| `default` | qdrant, redis-cache, langfuse-worker, langfuse-web, clickhouse, minio, redis, postgres |
| `cpu`     | vllm-cpu |
| `gpu`     | vllm |
| `compose` | rag-api, nginx |
| `canary`  | rag-api-canary |
| `mlflow`  | mlflow |

  ```bash
  docker-compose --profile cpu up
  docker-compose --profile cpu --profile mlflow up
  docker-compose --profile cpu --profile mlflow --profile compose up --build --force-recreate
  ```

- **`--profile gpu`** — adds the private `vllm` server (`Qwen/Qwen2.5-3B-Instruct-AWQ`),
  used instead of running vLLM locally. Needs an NVIDIA container runtime; the port
  is never published, `rag-api` reaches it at `http://vllm:8000/v1`.

- **NOTE**: Apple Silicon cannot GPU-accelerate the container: the `--profile gpu` service is for NVIDIA hosts only. To use the Apple GPU, run vLLM on the host with the community

  ```bash
  docker compose --profile gpu up -d
  ```

- **`--profile cpu`** — adds a CPU-only `vllm` (no GPU needed), served as the alias
  `vllm` so `rag-api` uses the same URL. AWQ is unsupported on the ARM CPU image
  (AWQ is x86/s390x/Power only), so it runs the unquantized
  `Qwen/Qwen3-0.6B` 

  ```bash
  docker-compose --profile cpu up -d
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
  docker-compose up qdrant mlflow # or local
  ```

* To evaluate the Query Path:
  ```bash
  uv run python -m ragapp.scripts.evaluate ask "ما هو مصدر التشريع"  --tracking-uri http://localhost:5000
  ```

* To evaluate the Offline Ingestion Path:
  ```bash
  dvc pull
  dvc repro --force
  uv run python -m ragapp.scripts.ingest_corpus
  ```


### Evaluation

* Runs every gold question through the RAG pipeline, captures the generated
answer plus the retrieved contexts, and scores the run in MLflow with RAGAS.

  ```bash
  uv run python -m ragapp.scripts.ragas --gold data/gold/gold_set.json \
          --judge-model <CHANGEME> --with-embeddings \
              --base-url http://localhost:8000/v1  --strict
  ```

* Also a debugging `Ragas` configuration is added to VSCode launch.json.


### Load testing

  ```bash
  uv run locust -f tests/load/locustfile.py \
    --headless -u 3 -r 2 -t 10s  \
    --host http://localhost:8000 \
    --html reports/smoke.html \
    --stop-timeout 90s
  ```

See report `reports/smoke.html` for performance.


## Canary Deployment
* A simple canary deployment is available through the nginx along with rag-api-canary in the docker-compose file. Traffic could be split between 2 backends wuth a fallback to the stable version.

  ```bahs
  docker-compose --profile cpu --profile compose --profile canary up --build --force-recreate
  ```

## Documentation

- `docs/PRD.md` — external brief.
- `docs/Architecture.md` — milestone plans.
<!-- - `docs/adr/` — architecture decision records. -->

