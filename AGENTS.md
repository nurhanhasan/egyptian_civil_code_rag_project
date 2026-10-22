# AGENTS.md — Egyptian Civil Code RAG

RAG system answering questions about the Egyptian Civil Code (Arabic legal Q&A).
Non-commercial educational use. Hallucinations are legally unacceptable — abstention
and citations are safety controls, not nice-to-haves.

## Setup

- Python `3.12.13` (see `.python-version`), managed with `uv` (`uv 0.12.x`).
- Install: `uv sync` (or `pip install -e .` must keep working — `src/` layout, import root `ragapp`).
- Never commit `.venv/`, `.ruff_cache/`, `__pycache__/`, `.env`, `mlflow/`, `mlflow.db`, `qdrant_storage/`, `test_mlflow_*.db`, `.DS_Store` (see `.gitignore`).
- There is no `docker-compose.yml` at the repo root; `docker/` holds `Dockerfile`, `Dockerfile-canary-nginx`, `nginx.template`. `monitoring/` is empty. `reports/` and `qdrant_storage/` contain local runtime outputs — do not assume shared infra exists; check before wiring to it.

## Commands

```bash
uv sync
uv run ruff check .        # lint (E, F, I)
uv run ruff format .        # format
uv run pytest tests/unit -q
uv run pytest tests/load -q   # Locust (no services required)

# Run FastAPI entrypoint (app is mounted under /api/v1)
docker-compose --profile cpu --profile mlflow up                  # deps without the api
uv run uvicorn ragapp.entrypoints.fastapi.main:app --no-access-log --reload
# then POST /api/v1/ask , GET /api/v1/health
```

- Line length: **88** (`pyproject.toml` `[tool.ruff]`). Lint selects `E, F, I` (pycodestyle, Pyflakes, isort).
- Pre-commit: `ruff-check --fix` + `ruff-format` (`.pre-commit-config.yaml`).
- Test layout: `tests/unit/` (fast deterministic), `tests/load/` (Locust). `tests/integration/` and `tests/e2e/` do not exist on disk yet.

## Source layout (`src/ragapp/`)

On-disk state:

```
src/ragapp/
  __init__.py  core/  dependencies/  entrypoints/fastapi/main.py  exceptions/  middleware/
  modules/
    health/          router.py service.py
    rag/
      __init__.py  dependency/rag.py
      domain/
        __init__.py  entities.py  ports.py  adapters/
          __init__.py embedders.py generators.py local_retriever.py rerankers.py retrievers.py
      evaluation/mlflow_evaluator.py
      factories.py  offline_ingestion/{__init__,ingestion,local_index}  router.py  schema.py  service.py
  scripts/  bge_ingest_corpus.py  qdrant_ingest_corpus.py  ragas.py  README.md
```

- `core/` = cache, config, context, logging, rate_limit. `dependencies/` = api_key, cache, rate_limit. `exceptions/` = domain, handlers, helpers. `middleware/` = cors, logging.
- There is NO top-level `domain/`, `adapters/`, `pipeline.py`, `repositories/`, or `modules/rag/tasks/` directory. Flattened plan (`src/ragapp/domain/`, `src/ragapp/adapters/`, `src/ragapp/pipeline.py`) is **Superseded**; the domain core (entities, ports) and adapters live under `modules/rag/domain/`; the pipeline orchestrates in `modules/rag/service.py` (`RAGPipelineService.run_pipeline`). Factory wiring is `modules/rag/factories.py` (single backend instantiation point).
- Import root is `ragapp`. Never `src.ragapp.*`.
- Prefer editing existing modules over adding new top-level packages.

## Binding architecture invariants

1. **Single public edge: FastAPI owns `POST /api/v1/ask`**. Routers are mounted under the `/api/v1` prefix (`main.py:98`). BentoML wraps embedder + reranker with adaptive batching only. vLLM is downstream, OpenAI-compatible HTTP, never imported as a library and never published.
2. **Exactly four ports** (`modules/rag/domain/ports.py`, all `Protocol`, all `async`): `Embedder`, `Retriever`, `Reranker`, `Generator`. Each port also declares `async close()`. A port earns its place with ≥2 implementations; current fakes + backends satisfy this. `Extractor`/`Chunker` are configured functions, NOT ports (single implementation each).
3. **Hybrid retrieval is the default**: `Embedder` returns dense AND sparse + `revision_id`. Two in-process `Retriever` arms now exist: Qdrant over the stable alias (default, `RETRIEVER_BACKEND=qdrant`) precompute index via `scripts/qdrant_ingest_corpus.py` into `qdrant_storage/` and BGE-M3 `RETRIEVER_BACKEND=bge`, precomputed index via `scripts/bge_ingest_corpus.py` into `bge_storage/`. No dense-only path except the baseline arm.
4. **Retrieval is never unfiltered.** `jurisdiction` + `doc_status` filters are mandatory parameters — make unfiltered retrieval unrepresentable in the signature (type error, not runtime check). Both retriever arms apply them before scoring.
5. **Two texts, always**: `raw_text` is the only text quoted in citations; `norm_text` is the only text embedded/matched; an offset map joins them. Arabic normalization toggles are config, logged per run.
6. **Citations carry `effective_date` + `doc_status` + `source_checksum`.** Quote `raw_text` only. Sort by score desc, cap at 5 (`MAX_CITATIONS` in `service.py`; M0 `Citation` schema #6, M1 #32).
7. **Abstention is the primary safety control** (M1 #33, four refusal conditions in `schema.py` docstring — SPEC.md is currently missing, do not invent them): no resolvable citation or low scores → `abstained=True` with reason and NO answer text (`AbstentionResponse.answer = None`). Do not weaken this without a superseding ADR.
8. **Qdrant immutable builds + alias promotion**: `/ask` reads the current alias (`QDRANT_COLLECTION_ALIAS`); ingestion is offline (CLI `scripts/ingest_corpus.py` / future `POST /admin/ingest`). Promotion/rollback = alias change, never image rebuild.
9. **Two seams only** : an `Evaluator` *instance* (constructed, passed explicitly — no globals/class-mutable state; `modules/rag/evaluation/mlflow_evaluator.py`) and OpenTelemetry directly. **Import lint: `pipeline.py` (planned) and `domain/**` must never `import mlflow` or `celery`** (enforced by `tests/unit/test_import_lint.py`). MLflow is allowed in `evaluation/` only. OTel is permitted in pipeline/domain (it IS the neutral layer); exporters default to no-op — app serves with observability down. Current request-path tracing uses Langfuse observations (`service.py`).
10. **Eval tiers**: CI blocks on deterministic Tier-1 metrics only; RAGAS alerts, never blocks. Old `faithfulness ≥ 0.75` blocking gate is retired — do not reintroduce it.
11. **Config is two-tier** : environment (secrets, URIs) vs `params.yaml` (chunking, retrieval, model revisions). Log the full param set per run so M2 comparisons are attributable. `params.yaml` is committed; Qdrant URIs etc. set via env in CI.

## API contract

- `POST /api/v1/ask`: `{question: str}` (`min_length=1`) → `{answer: str, sources: [...]}` or `{abstained: true, reason: str, answer: null, sources: []}`. Empty question → **422** via Pydantic, tested. Trace ID returned in `X-Trace-ID` header.
- `GET /api/v1/health` → `{status: healthy, documents_indexed: N}`.
- Span tree per `/ask`: `ask → embedding → retrieving → reranker → generate → citation` (Langfuse observations in `service.py`).

## Tests — what good looks like

- Fakes are **deterministic** (same input → byte-identical output) and emit **Arabic** canned text so UTF-8/RTL breaks early.
- Cover: ports satisfied by fakes; pipeline determinism across runs; `Evaluator` concurrency (two instances, interleaved `log_metrics`, no cross-talk); OTel/Langfuse no-op (unset endpoint → serves, no error); span tree via in-memory exporter; abstention path (`abstained=True` + reason, no answer text); import-lint negative test.
- `tests/unit/`: test_evaluator_concurrency, test_evaluator_params_logged, test_generator, test_import_lint, test_local_retriever, test_ragas_mlflow_logging, test_rate_limit, test_retriever. `tests/load/locustfile.py`.
- New behavior needs a unit test; retrieval/citation changes need integration coverage with mandatory-filter assertions (leakage = 0).

## Docs discipline

- `docs/adr/` records are **immutable once Accepted** — supersede (`Superseded by ADR-NNNN`), never edit. New decisions take the next free number (currently 0018) with Context/Decision/Consequences/Alternatives-rejected.
- Plans live in `docs/implementation_plans/` (template §1–7) and **cite ADR numbers + SPEC sections** — they never restate decisions. ADR wins on conflict.
- `docs/Project_Details.md` is the external brief — never edit it.
- `SPEC.md` and `MLOPS_TODO.md` are referenced but missing from the repo root (`docs/mlops_plan.md` exists instead) — say so instead of guessing their contents.

## Guardrails for agents

- Do: keep `domain/` pure (dataclasses, Protocols, policies); inject deps via ports; use `uv run`; run ruff check + format before finishing; prefer `modules/rag/domain/` for ports/entities (no top-level `domain/`).
- Do not: add a 5th port / new seam; import `mlflow`/`celery`/vLLM libs into pipeline or `domain/**`; return answers without resolvable citations; chart pre-M3 CPU latency against target-hardware figures; invent SPEC.md/MLOPS_TODO.md content that isn't on disk.
- If `SPEC.md` or `MLOPS_TODO.md` is referenced but missing from the repo root, say so instead of guessing its contents.
