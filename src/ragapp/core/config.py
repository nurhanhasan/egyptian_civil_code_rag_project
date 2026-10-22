import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_env_file():
    """
    Dynamically determine the path to the .env file based on the environment.
    Usage:
    Available values:
    dev, test, prod
    """
    env = os.getenv("ENVIRONMENT", "dev")
    return f".env.{env}"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables take precedence over values defined here.

    A .env file can be used for local development.
    """

    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_DEBUG: bool = True
    APP_NAME: str = "RAGApp"
    ENVIRONMENT: str

    # Logging
    LOGGING_LEVEL: str

    # Redis / Cache — MLOPS_TODO #8
    #
    # One Redis instance, four SEPARATE logical databases.
    #   cache     (db 0) — ephemeral lookups
    #   broker    (db 1) — Celery broker           (M4 #83)
    #   result    (db 2) — Celery result backend   (M4 #83)
    #   ratelimit (db 3) — distributed token bucket (M0 #9)
    REDIS_CACHE_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND_URL: str = "redis://localhost:6379/2"
    REDIS_RATELIMIT_URL: str = "redis://localhost:6379/3"

    # Generic healthcheck/ops ping target. Defaults to the cache db.
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_NAMESPACE: str
    CACHE_TTL: int = 60

    # CORS
    CORS_ORIGINS: List[str]

    # Static API key. Flag-gated, off by default in local dev.
    API_KEY: str | None = None
    API_KEY_ENABLED: bool = False

    # LLM Generator
    #   fake    (default) — deterministic, offline; CI + unit tests + GPU-free dev
    #   openai           — any OpenAI-compatible /v1/chat/completions server:
    #                       vLLM in prod (Qwen2.5-7B-AWQ), Ollama in dev,
    #                       TGI / llama.cpp / Azure — only the base_url+model

    GENERATOR_BACKEND: str = "openai"
    GENERATOR_SYSTEM_PROMPT: str = (
        "You are a careful assistant that answers questions about the Egyptian "
        "Civil Code. Answer only from the context provided. If the context does "
        "not contain the answer, say so explicitly instead of guessing. "
        "When the user asks in Arabic, always answer in Arabic. "
        "Only if the user asks in English, answer in English."
    )
    OPENAI_BASE_URL: str | None = None
    OPENAI_API_KEY: str | None = None
    GENERATOR_MODEL: str = ""
    GENERATOR_TEMPERATURE: float = 0.0
    GENERATOR_MAX_TOKENS: int = 1000
    GENERATOR_TIMEOUT: float = 120.0

    ### BGE Device
    # None = auto (CUDA > MPS > CPU). Set "cpu" to force CPU, e.g. macOS local
    # dev where concurrent MPS access trips a Metal assertion.
    BGE_DEVICE: str | None = None

    ### Embedder ###
    EMBEDDER_BACKEND: str = "bge"
    # Model paths for local cached models
    BGE_M3_MODEL_PATH: str = "BAAI/bge-m3"
    EMBEDDER_MODEL_REVISION_ID: str = "5617a9f61b028005a4858fdac845db406aefb181"

    ### Reranker ###
    RERANKER_BACKEND: str = "bge"
    # Model paths for local cached models
    BGE_RERANKER_MODEL_PATH: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MODEL_REVISION_ID: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    # Retriever — Qdrant. URIs + the stable alias live in the
    # environment tier; top_k / RRF-k belong to the params.yaml tier.
    RETRIEVER_BACKEND: str = "qdrant"
    QDRANT_URL: str = "http://localhost:6333"
    # When set, use Qdrant client local mode (in-process engine) over this
    # directory instead of an HTTP server.
    QDRANT_RETRIEVER_PATH: str | None = None
    QDRANT_COLLECTION_ALIAS: str = "egyptian-civil-code"
    QDRANT_TIMEOUT: int = 5

    # Local (in-process) BGE-M3 retriever arm — RETRIEVER_BACKEND=bge.
    # Reads the precomputed index built by `ragapp.scripts.bge_ingest_corpus`.
    BGE_RETRIEVER_PATH: str = "bge_storage"
    RETRIEVER_RRF_K: int = 60

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = False
    # Tokens added to each key's bucket per REFILL_PERIOD seconds.
    RATE_LIMIT_CAPACITY: int = 60
    RATE_LIMIT_REFILL_PERIOD: int = 60
    # Bucket TTL so idle keys don't pile up in Redis (>= 2 x REFILL_PERIOD).
    RATE_LIMIT_BUCKET_TTL: int = 120

    # MLflow
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT_NAME: str = "egyptian-civil-code-rag"

    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_BASE_URL: str = "http://localhost:3000"

    # params.yaml path
    PARAMS_YAML_PATH: str = "params.yaml"

    def load_params(self) -> dict[str, Any]:
        """Load params.yaml and merge with settings for Evaluator.

        Returns flat dict with all 7 categories:
        chunking, embedding, retrieval, reranking, generation, corpus/index, evaluation.
        """
        params_path = Path(self.PARAMS_YAML_PATH)
        yaml_params: dict[str, Any] = {}
        if params_path.exists():
            with params_path.open("r", encoding="utf-8") as f:
                yaml_params = yaml.safe_load(f) or {}

        # Normalization toggles from params.yaml
        normalization_toggles = {
            "no_decompose_presentation_forms": yaml_params.get(
                "no_decompose_presentation_forms", False
            ),
            "no_strip_tatweel": yaml_params.get("no_strip_tatweel", False),
            "no_strip_tashkeel": yaml_params.get("no_strip_tashkeel", False),
            "no_unify_letters": yaml_params.get("no_unify_letters", False),
            "no_fold_digits": yaml_params.get("no_fold_digits", False),
            "no_collapse_whitespace_punct": yaml_params.get(
                "no_collapse_whitespace_punct", False
            ),
        }

        # Determine tracking URI with sqlite fallback
        tracking_uri = self.MLFLOW_TRACKING_URI or "sqlite:///mlflow.db"

        return {
            # Chunking
            "chunking_strategy": yaml_params.get("strategy", "article-aware"),
            "chunk_size": yaml_params.get("fixed_size_chars", 1200),
            "chunk_overlap": yaml_params.get("fixed_overlap_chars", 200),
            "normalization_toggles": normalization_toggles,
            # Embedding
            "embedder_backend": self.EMBEDDER_BACKEND,
            "embedder_model_id": self.BGE_M3_MODEL_PATH,
            "embedder_revision_id": self.EMBEDDER_MODEL_REVISION_ID,
            # Retrieval
            "retriever_backend": self.RETRIEVER_BACKEND,
            "top_k": 5,  # hardcoded in service.py for now
            "rrf_k": self.RETRIEVER_RRF_K,
            "retrieval_mode": "hybrid",
            "jurisdiction": "EG",
            "doc_status": "in_force",
            # Reranking
            "reranker_backend": self.RERANKER_BACKEND,
            "reranker_model_id": self.BGE_RERANKER_MODEL_PATH,
            "reranker_revision_id": self.RERANKER_MODEL_REVISION_ID,
            # Generation
            "generator_backend": self.GENERATOR_BACKEND,
            "generator_model": self.GENERATOR_MODEL,
            "generator_revision": "",
            "temperature": self.GENERATOR_TEMPERATURE,
            "max_tokens": self.GENERATOR_MAX_TOKENS,
            "system_prompt_hash": str(hash(self.GENERATOR_SYSTEM_PROMPT)),
            # Corpus/Index
            "corpus_revision": "",  # filled at ingestion time (DVC hash)
            "index_build_id": "",  # filled at ingestion time (Qdrant alias target)
            # Evaluation
            "judge_model": "",
            "judge_version": "",
            "gold_set_version": "",
            "eval_tier": 1,
            # MLflow
            "mlflow_tracking_uri": tracking_uri,
            "mlflow_experiment_name": self.MLFLOW_EXPERIMENT_NAME,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    """
    return Settings()
