# Arabic Legal Document Q&A -- Project 2
###### Arabic legal documents · Embeddings + Vector DB · vLLM · RAGAS · Langfuse · Guardrails

- [Arabic Legal Document Q\&A -- Project 2](#arabic-legal-document-qa----project-2)
          - [Arabic legal documents · Embeddings + Vector DB · vLLM · RAGAS · Langfuse · Guardrails](#arabic-legal-documents--embeddings--vector-db--vllm--ragas--langfuse--guardrails)
  - [The business problem](#the-business-problem)
  - [Why this project for this course](#why-this-project-for-this-course)
  - [What you ship](#what-you-ship)
  - [Completion checklist](#completion-checklist)
    - [CODE \& PACKAGING](#code--packaging)
    - [EXPERIMENT TRACKING \& DATA VERSIONING](#experiment-tracking--data-versioning)
    - [SERVING \& RELEASE](#serving--release)
    - [MONITORING \& OBSERVABILITY](#monitoring--observability)
    - [MODEL OPTIMIZATION](#model-optimization)


## The business problem
A law firm or government authority in the Arab world needs a system that answers questions about Arabic legal documents — contracts, Egyptian Civil Code articles, Saudi regulations — accurately and with citations. Hallucinations are legally unacceptable.


## Why this project for this course
— RAGAS evaluation is genuinely hard — deep S4 content
— No GPU required to start — API-based LLM works fine
— Embedding drift is real — legal language shifts across jurisdictions
— Langfuse tracing reveals the full RAG chain
— AWQ quantization in S5 makes the LLM run fully offline


## What you ship
Package + API   : src/rag.py (ingest+query), /ask: {"question":"ما هي شروط العقد؟"} → {"answer":..., "sources":[...]}
Track + Version : MLflow: chunking experiments (chunk_size, overlap, RAGAS faithfulness). DVC: document corpus.
Serve           : BentoML wraps RAG. vLLM serves generative model. Streaming /ask.
Monitor         : RAGAS on ≥ 50 questions. Langfuse traces every request. Cosine drift. Token cost.
Optimize        : AWQ-4bit generative model. Re-ranker distillation. RAGAS before vs after.


## Completion checklist

### CODE & PACKAGING

- [   ] src/ layout with pyproject.toml — pip install -e . works.
- [   ] Ingestion pipeline: PDF/text → chunk → embed → vector store.
- [   ] FastAPI /ask: {question: str} → {answer: str, sources: list[str]}
- [   ] Pydantic rejects empty question — 422 returned and tested
- [   ] /health returns {status: healthy, documents_indexed: N}
- [   ] Dockerfile includes vector store and embedded documents
- [   ] docker compose up starts service on port 8000
- [   ] README: 3 commands to run Q&A on any machine


### EXPERIMENT TRACKING & DATA VERSIONING

- [   ] MLflow logs each config: chunk_size, overlap, embedding_model, faithfulness
- [   ] ≥ 5 runs compared — MLflow screenshot in /reports/
- [   ] Best chunking config registered in MLflow Registry
- [   ] Document corpus tracked with DVC — dvc pull fetches all source docs
- [   ] dvc repro re-indexes corpus reproducibly from tracked documents
- [   ] GitHub Actions: lint → test → rebuild index → push Docker image
- [   ] CI fails if RAGAS faithfulness < 0.75 on 20-question test set


### SERVING & RELEASE

- [   ] BentoML wraps RAG pipeline with async /ask endpoint
- [   ] vLLM serves the generative LLM — model name in README
- [   ] Streaming: tokens appear progressively in curl output
- [   ] Locust report at 50 concurrent users in /reports/
- [   ] Batch re-indexing script tested with at least one new document. (async Celery ingestion?!)
- [   ] Canary rollout config documented in README


### MONITORING & OBSERVABILITY

- [   ] RAGAS on ≥ 50 questions — all 4 metrics logged
- [   ] RAGAS results stored in MLflow — trend visible across trials
- [   ] Alert: faithfulness < 0.80 → notification triggered
- [   ] Langfuse self-hosted — every /ask creates a trace with spans
- [   ] RAGAS faithfulness score attached to each Langfuse trace
- [   ] Cosine similarity on query embeddings vs baseline — drift logged
- [   ] Token usage in Prometheus — Grafana shows cost/hour
- [   ] Guardrails PII detection active on all /ask responses


### MODEL OPTIMIZATION

- [   ] Generative model quantized to AWQ-4bit — weights documented
- [   ] RAGAS scores: original vs quantized — ≤ 0.03 faithfulness drop
- [   ] Latency before vs after quantization in README
- [   ] Final architecture diagram covering all components
