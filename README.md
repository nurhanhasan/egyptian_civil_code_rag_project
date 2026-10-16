# Egyptian Civil Code RAG

- [Egyptian Civil Code RAG](#egyptian-civil-code-rag)
  - [Run Q\&A in 3 commands](#run-qa-in-3-commands)
  - [Run without Docker (development)](#run-without-docker-development)
  - [Endpoints](#endpoints)
  - [Models and revisions](#models-and-revisions)
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

  ```bash
  # 1. Clone
  git clone https://github.com/nurhanhasan/egyptian_civil_code_rag_project && cd egyptian_civil_code_rag_project && cp .env.example .env.dev # edit, if needed.

  # 2. Start CPU profile. For GPU, pick the `gpu` one.
  docker-compose --profile cpu up --build -d

  # 3. Ask a question
  curl -X 'POST' \
    'http://localhost/api/v1/ask' \
    -H 'accept: */*' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "متى تسقط الدعوى المدنية بالتقادم؟"
  }'


  # And, for abstention

  curl -X 'POST' \
    'http://localhost/api/v1/ask' \
    -H 'accept: */*' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "ما عقوبة جريمة القتل"             
  }'

  # finally,
  docker-compose --profile cpu down
  ```

* The API is served on <http://localhost:8000>. Interactive docs: [/docs](http://localhost:8000/docs).

* To build the index
  ```bash
  docker-compose exec rag-api python -m ragapp.scripts.ingest_corpus
  ```

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


## Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/api/v1/ask` | `{"question": str}` | `{"answer": str, "sources": [...]}` or an abstention `{"abstained": true, "reason": str, "answer": null, "sources": []}` |
| `GET` | `/api/v1/health` | — | `{"status": "ok"}` |

An empty or missing `question` is rejected with `422` (Pydantic validation).


## Models and revisions

| Role | Model | Revision |
|---|---|---|
| Embedder | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| Reranker | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Generator | `Qwen/Qwen3-0.6B` | served via vLLM |

All three are configurable through the environment (`BGE_M3_MODEL_PATH`,
`BGE_RERANKER_MODEL_PATH`, `GENERATOR_MODEL`). Pinning an immutable revision per
model is required.


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

* For the MLOps Maturity breakdown, visit the branches in this order
  1. [no-mlops](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/no-mlops)
  2. [devops-but-not-mlops](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/devops-but-not-mlops)
  3. [mlops-w-atomated-data-pipeline](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/mlops-w-atomated-data-pipeline)
  4. [mlops-automated-deployment](https://github.com/nurhanhasan/egyptian_civil_code_rag_project/tree/mlops-automated-deployment)


## Compose Profiles

- **default** — `rag-api`, `qdrant`, `redis`, `nginx` & `langfuse` stack.

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

- Runs every gold question through the RAG pipeline, captures the generated
answer plus the retrieved contexts, and scores the run in MLflow with RAGAS.

  ```bash
  uv run python -m ragapp.scripts.ragas --gold data/gold/gold_set.json \
          --judge-model <CHANGEME> --with-embeddings \
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

See report `reports/smoke.html` for performance.


## Documentation

- `docs/PRD.md` — external brief.
- `docs/Architecture.md` — milestone plans.
<!-- - `docs/adr/` — architecture decision records. -->

