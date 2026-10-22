# Egyptian Civil Code RAG

- [Egyptian Civil Code RAG](#egyptian-civil-code-rag)
  - [Quickstart](#quickstart)
    - [Run Q\&A in 3 commands](#run-qa-in-3-commands)
    - [Run without Docker (development)](#run-without-docker-development)
    - [Endpoints](#endpoints)
    - [Compose Profiles](#compose-profiles)
  - [How it works](#how-it-works)
    - [Components](#components)
    - [Models and revisions](#models-and-revisions)
    - [Project layout](#project-layout)
    - [Experimental tracking](#experimental-tracking)
    - [DVC data versioning and reproducibility](#dvc-data-versioning-and-reproducibility)
    - [Evaluation](#evaluation)
    - [Load testing](#load-testing)
  - [Canary Deployment](#canary-deployment)
  - [Documentation](#documentation)


A Retrieval-Augmented Generation (RAG) system that answers questions about the
Egyptian Civil Code (Arabic legal Q&A) with citations. Non-commercial educational
use. Hallucinations are legally unacceptable here, so abstention and citations are
safety controls.



## Quickstart

### Run Q&A in 3 commands

  NOTE: To be able to run this locally, make sure to allocate at least 4 Cores & 16GB of memory.

  NOTE: `rag-api` keeps the Hugging Face cache in the `hf-cache` named Docker volume. The first start downloads BGE-M3 + the reranker (~4.5 GB, plus the CPU generator) and blocks in the startup pre-warm until they load, so `/api/v1/health` stays red for several minutes. Watch `docker compose logs -f rag-api` for `Uvicorn running` before calling the API. Do not recreate the container mid-download; `docker compose down` keeps the volume, `down -v` deletes it.

  NOTE: `rag-api` is pinned to `platform: linux/amd64` and `pull_policy: always`. On Apple Silicon it runs under QEMU (slow builds and model load). For a native run, comment out both `platform` and `pull_policy` in `docker-compose.yaml`, then `docker compose build rag-api`.

  ```bash
  # 1. Clone
  git clone https://github.com/nurhanhasan/egyptian_civil_code_rag_project && cd egyptian_civil_code_rag_project && cp .env.example .env.dev  # edit, if needed.

  # 2. Start CPU profile. For GPU, pick the `gpu` one.
  docker compose --profile cpu --profile compose up -d

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
  docker compose --profile cpu --profile mlflow --profile compose down
  ```

* The API is served on <http://localhost:8000>. Interactive docs: [/docs](http://localhost:8000/docs).

* For artifacts and screenshots, please check [Here](./reports/README.md).

* To build the index and bake vector store in the docker image:
  ```bash
  uv run python -m ragapp.scripts.ingest_corpus --qdrant-path qdrant_storage
  docker compose build rag-api

  # `pull_policy: always` would replace the local build on the next `up`;
  # `--pull never` keeps the image you just built.
  docker compose --profile cpu --profile compose up -d --pull never
  ```

### Run without Docker (development)

  ```bash
  uv sync

  docker compose --profile cpu --profile mlflow up                  # deps without the api
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
| `default` | redis-cache, langfuse-worker, langfuse-web, clickhouse, minio, redis, postgres |
| `cpu`     | vllm-cpu |
| `gpu`     | vllm |
| `compose` | rag-api, nginx |
| `canary`  | rag-api-canary |
| `mlflow`  | mlflow |

  ```bash
  docker compose --profile cpu up
  docker compose --profile cpu --profile mlflow up
  docker compose --profile cpu --profile mlflow --profile compose up --build --force-recreate
  ```

- **`--profile gpu`** — adds the private `vllm` server (`Qwen/Qwen2.5-3B-Instruct-AWQ`),
  used instead of running vLLM locally. Needs an NVIDIA container runtime; the port
  is never published, `rag-api` reaches it at `http://vllm:8000/v1`.

- **NOTE**: Apple Silicon cannot GPU-accelerate the container: the `--profile gpu`
  service is for NVIDIA hosts only. To use the Apple GPU, run vLLM on the host
  with a Metal build and point `OPENAI_BASE_URL` at it.

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
  Redis, vLLM and the Langfuse stores live on the private `appnet` network.
  Qdrant runs in-process (`QDRANT_RETRIEVER_PATH`), so there is no vector-store service
  or port; local development still publishes dependency ports for convenience.

* **Immutable builds, alias promotion.** Ingestion creates a new collection named
  by its build ID, populates it, then moves the stable alias
  (`egyptian-civil-code`) to it. `/ask` reads only the alias. Promotion and
  rollback are alias changes — never an image rebuild or a mutation of a live
  collection. In `QDRANT_RETRIEVER_PATH` (local) mode the alias lives in `meta.json`;
  server-format storage and local-mode storage are not interchangeable.

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
  scripts/         bge_ingest_corpus.py, qdrant_ingest_corpus.py, ragas.py
tests/
  unit/ integration/ e2e/ load/
docs/              PRD, Architecture, ADRs, implementation plans
data/corpus/       source + processed corpus (DVC-tracked)
```

* For the MLOps Maturity breakdown, visit the branches in this order
  1. [no-mlops](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/no-mlops)
  2. [devops-but-not-mlops](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/devops-but-not-mlops)
  3. [mlops-w-atomated-data-pipeline](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/mlops-w-atomated-data-pipeline)
  4. [mlops-automated-deployment](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/mlops-automated-deployment)


<!-- ## Quality gates -->

<!-- TODO: -->
<!-- In progress ... -->


<!-- ## Run, Track & Evaluate -->

### Experimental tracking

To track experiments:

  ```bash
  docker compose --profile mlflow up # or local
  ```

* To evaluate the Query Path:
  ```bash
  uv run python -m ragapp.scripts.evaluate ask "ما هو مصدر التشريع"  --tracking-uri http://localhost:5000
  ```

* To evaluate the Offline Ingestion Path (local-mode store, no server):
  ```bash
  uv run dvc pull
  uv run dvc repro
  uv run python -m ragapp.scripts.ingest_corpus --qdrant-path qdrant_storage
  ```

### DVC data versioning and reproducibility

DVC versions the corpus and the derived extraction/chunking outputs. Git only
stores the `.dvc` pointers plus `dvc.yaml` / `dvc.lock`, so a clone can restore
the exact same inputs and derived artifacts.

| Tracked by DVC | Path |
|---|---|
| Source PDF | `data/corpus/Egyptian_Civil_Law.pdf` (via `Egyptian_Civil_Law.pdf.dvc`) |
| Extracted document | `data/corpus/processed/document.json` |
| Page text | `data/corpus/processed/pages.jsonl` |
| Article-aware chunks | `data/corpus/processed/chunks.article_aware.jsonl` |
| Source-to-chunk map | `data/corpus/processed/source_to_chunk.jsonl` |
| Validation report | `data/corpus/processed/validation_report.json` |

The `extract-and-chunk` stage in `dvc.yaml` is driven by `params.yaml`
(`pdf`, `out_dir`, `strategy`) and depends on the PDF and
`data/corpus/extract_civil_code.py`. Changing either a parameter or a
dependency invalidates the stage. `data/corpus/processed/` is gitignored —
never commit its contents.

Two remotes are configured:

```bash
# ❯ uv run dvc remote list
ecc-doc-link    https://sadanykhalifa.com/uploads/Laws/1576751803.pdf
ecc-dvc-data-s3 s3://ecc-dvc-data       (default)
```

`ecc-dvc-data-s3` is the DVC cache used by `pull`/`push`. `ecc-doc-link`
records the upstream source URL. `dvc pull` can read from the public bucket;
`dvc push` needs AWS credentials with write access.

In CI, the `rebuild-index` job in `.github/workflows/ci.yml` runs
`dvc pull` → `dvc repro` → `dvc push`, then builds the vector store with
`ingest_corpus --qdrant-path qdrant_storage`. Qdrant client local mode writes
it (`meta.json` + per-collection SQLite), so no Qdrant server is involved.
The job uploads `qdrant_storage/` as an artifact; the `build` job bakes it
into the API image at `/app/qdrant_storage`, and `QDRANT_RETRIEVER_PATH` makes the API
serve it in-process. The job therefore needs both read and write AWS
credentials for the S3 remote that's configured through Environment Secrets.

  ```bash
  uv sync

  uv run dvc pull        # restore the PDF and cached processed outputs
  uv run dvc repro       # re-run extract-and-chunk only if deps/params changed
  uv run dvc push        # upload new outputs to the default remote

  # Build the local-mode vector store (no server; writes qdrant_storage/)
  uv run python -m ragapp.scripts.ingest_corpus --qdrant-path qdrant_storage

  # Bake qdrant_storage/ into the API image (QdrantClient local mode)
  docker compose build rag-api
  ```


### Evaluation

* Runs every gold question through the RAG pipeline, captures the generated
answer plus the retrieved contexts, and scores the run in MLflow with RAGAS.

  ```bash
  uv run python -m ragapp.scripts.ragas --gold data/gold/gold_set.json \
          --judge-model <CHANGEME> --with-embeddings \
              --base-url http://localhost:8000/v1  --strict
  ```

  `--base-url` is the judge endpoint: use `http://localhost:7080/v1` with the
  `cpu` profile, `http://localhost:8000/v1` with `gpu`.

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
* A simple canary deployment is available through the nginx along with rag-api-canary in the docker compose file. Traffic could be split between 2 backends wuth a fallback to the stable version.

  ```bahs
  docker compose --profile cpu --profile compose --profile canary up --build --force-recreate
  ```

## Documentation

- `docs/PRD.md` — external brief.
- `docs/Architecture.md` — milestone plans.
<!-- - `docs/adr/` — architecture decision records. -->

