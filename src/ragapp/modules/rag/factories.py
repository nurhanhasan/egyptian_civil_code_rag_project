"""
`factories.py` is the single selection point that decides WHICH implementation
the pipeline gets, driven purely by config:
"""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from ragapp.core.config import Settings
from ragapp.modules.rag.domain.adapters.embedders import (
    BentoMLBGEEmbedder,
    _FakeEmbedder,
)
from ragapp.modules.rag.domain.adapters.generators import _FakeGenerator, _OpenAIBackend
from ragapp.modules.rag.domain.adapters.local_retriever import BgeM3Retriever
from ragapp.modules.rag.domain.adapters.rerankers import (
    BentoMLBGEReranker,
    _FakeReranker,
)
from ragapp.modules.rag.domain.adapters.retrievers import QdrantRetriever
from ragapp.modules.rag.domain.ports import Embedder, Generator, Reranker, Retriever

logger = logging.getLogger(__name__)


_SUPPORTED_BACKENDS = ("fake", "openai")


@lru_cache(maxsize=1)
def _get_generator_cached(
    backend: str,
    openai_base_url: str | None,
    openai_api_key: str | None,
    generator_model: str,
    generator_temperature: float,
    generator_max_tokens: int | None,
    generator_timeout: float,
) -> Generator:
    """
    Why generato cached?
    _OpenAIBackend creates an openai.OpenAI client with connection pooling.
    Reusing the same instance avoids recreating connection pools.

    reset_generator_cache() for test isolation, clears the cache between
    tests for isolation.
    """
    logger.info(f"Creating generator: backend={backend}")

    if backend == "fake":
        logger.info("Using _FakeGenerator")
        return _FakeGenerator()

    if backend == "openai":
        logger.info("Using _OpenAIBackend")
        logger.info(f"OpenAI Base URL: {openai_base_url}, Model: {generator_model}")
        if not openai_base_url:
            raise ValueError(
                "GENERATOR_BACKEND=openai requires OPENAI_BASE_URL "
                "(e.g. http://vllm:8000/v1 or http://localhost:11434/v1)"
            )
        gen: Generator = _OpenAIBackend(
            base_url=openai_base_url,
            model=generator_model,
            api_key=openai_api_key or "",
            temperature=generator_temperature,
            max_tokens=generator_max_tokens,
            timeout=generator_timeout,
        )
        return gen

    raise ValueError(
        f"Unsupported GENERATOR_BACKEND={backend!r}. "
        f"Expected one of: {', '.join(_SUPPORTED_BACKENDS)}"
    )


def get_generator(settings: Settings) -> Generator:
    """
    Build the configured Generator.

        GENERATOR_BACKEND=fake    -> _FakeGenerator     (CI / unit / GPU-free)
        GENERATOR_BACKEND=openai  -> _OpenAIBackend     (any OpenAI-compatible server)
    """
    backend = (settings.GENERATOR_BACKEND or "fake").strip().lower()
    return _get_generator_cached(
        backend=backend,
        openai_base_url=settings.OPENAI_BASE_URL,
        openai_api_key=settings.OPENAI_API_KEY,
        generator_model=settings.GENERATOR_MODEL,
        generator_temperature=settings.GENERATOR_TEMPERATURE,
        generator_max_tokens=settings.GENERATOR_MAX_TOKENS,
        generator_timeout=settings.GENERATOR_TIMEOUT,
    )


def reset_generator_cache() -> None:
    """Clear the generator factory cache (used by tests)."""
    _get_generator_cached.cache_clear()


@lru_cache(maxsize=1)
def _get_embedder_cached(backend: str, device: str | None) -> Embedder:

    logger.info(f"Creating embedder: backend={backend}, device={device}")

    if backend == "fake":
        logger.info("Using _FakeEmbedder")
        return _FakeEmbedder()

    logger.info("Using BentoMLBGEEmbedder")
    # `.inner` is BentoML's raw class — the seam for injecting constructor args;
    # the @service() wrapper's zero-arg call stays available for the runtime.
    return BentoMLBGEEmbedder.inner(device=device)


def get_embedder(settings: Settings) -> Embedder:
    return _get_embedder_cached(settings.EMBEDDER_BACKEND, settings.BGE_DEVICE)


def reset_embedder_cache() -> None:
    """Clear the embedder factory cache (used by tests)."""
    _get_embedder_cached.cache_clear()


@lru_cache(maxsize=1)
def _get_retriever_cached(
    backend: str,
    collection_alias: str,
    url: str | None,
    timeout: int | None,
    index_path: str,
    rrf_k: int,
) -> Retriever:
    """
    Build the configured Retriever.

        RETRIEVER_BACKEND=qdrant -> QdrantRetriever over QDRANT_URL,
                                    reading the stable alias
                                    QDRANT_COLLECTION_ALIAS
        RETRIEVER_BACKEND=bge    -> BgeM3Retriever over a precomputed local
                                    index at RETRIEVER_INDEX_PATH

    QDRANT_URL=":memory:" selects Qdrant's in-process local store —
    hermetic, GPU/docker-free CI and local dev.
    """

    logger.info(
        f"Creating retriever: backend={backend}, url={url}, alias={collection_alias}"
    )

    if backend == "qdrant":
        url = url
        if url == ":memory:":
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(url=url, timeout=timeout)

        return QdrantRetriever(client=client, collection_alias=collection_alias)

    if backend == "bge":
        logger.info(f"Using BgeM3Retriever over local index: {index_path}")
        return BgeM3Retriever(index_path, rrf_k=rrf_k)

    raise ValueError(
        f"Unsupported RETRIEVER_BACKEND={backend!r}. Expected: qdrant, bge"
    )


def get_retriever(settings: Settings) -> Retriever:
    return _get_retriever_cached(
        backend=settings.RETRIEVER_BACKEND,
        url=settings.QDRANT_URL,
        collection_alias=settings.QDRANT_COLLECTION_ALIAS,
        timeout=settings.QDRANT_TIMEOUT,
        index_path=getattr(
            settings, "RETRIEVER_INDEX_PATH", "data/corpus/processed/local_index"
        ),
        rrf_k=getattr(settings, "RETRIEVER_RRF_K", 60),
    )


@lru_cache(maxsize=1)
def _get_reranker_cached(backend: str, device: str | None) -> Reranker:

    logger.info(f"Creating reranker: backend={backend}, device={device}")

    if backend == "fake":
        logger.info("Using _FakeReranker")
        return _FakeReranker()

    logger.info("Using BentoMLBGEReranker")
    return BentoMLBGEReranker.inner(device=device)


def get_reranker(settings: Settings) -> Reranker:
    return _get_reranker_cached(settings.RERANKER_BACKEND, settings.BGE_DEVICE)
