"""FastAPI dependency for the RAG service and its dependencies."""

from __future__ import annotations

from fastapi import Depends

from ragapp.core.config import Settings, get_settings
from ragapp.modules.rag.domain.ports import Embedder, Generator, Reranker, Retriever
from ragapp.modules.rag.factories import (
    get_embedder,
    get_generator,
    get_reranker,
    get_retriever,
)
from ragapp.modules.rag.service import RAGPipelineService


def get_generator_dep(settings: Settings) -> Generator:
    return get_generator(settings)


def get_embedder_dep(settings: Settings) -> Embedder:
    return get_embedder(settings)


def get_retriever_dep(settings: Settings) -> Retriever:
    return get_retriever(settings)


def get_reranker_dep(settings: Settings) -> Reranker:
    return get_reranker(settings)


def get_rag_pipeline_service(
    settings: Settings = Depends(get_settings),
) -> RAGPipelineService:
    """
    TODO: Document and move the below docstring as needed.

    The three legitimate ways to get a bentoml service instance;
      1. RAGService(): requires __init__(self) zero-arg; adds .to_sync/.to_async
      2. RAGService.inner(generator=..., ...): inner is your raw class →
            your kwargs work; no to_sync/to_async,
            but plain run_pipeline() still works
            (APIMethod.__get__ returns a local caller, method.py:143)
      3. bentoml.depends(RAGService): class attr of another @bentoml.service;
         .get() returns in-process self.on() when
         the name isn't in remote_runner_mapping,
         else a RemoteProxy over HTTP (dependency.py:73-84) —
         this is the seam BentoML designed for
    """
    return RAGPipelineService(
        generator=get_generator_dep(settings),
        embedder=get_embedder_dep(settings),
        retriever=get_retriever_dep(settings),
        reranker=get_reranker_dep(settings),
        system_prompt=settings.GENERATOR_SYSTEM_PROMPT,
    )
