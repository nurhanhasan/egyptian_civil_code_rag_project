"""
`factories.py` is the single selection point that decides WHICH implementation
the pipeline gets, driven purely by config:
"""

from __future__ import annotations

import logging

from ragapp.core.config import Settings
from ragapp.modules.rag.domain.adapters.generators import _FakeGenerator, _OpenAIBackend
from ragapp.modules.rag.domain.ports import Embedder, Generator, Reranker, Retriever

logger = logging.getLogger(__name__)


_SUPPORTED_BACKENDS = ("fake", "openai")


def get_generator(settings: Settings) -> Generator:
    """
    Build the configured Generator.

        GENERATOR_BACKEND=fake    -> _FakeGenerator     (CI / unit / GPU-free)
        GENERATOR_BACKEND=openai  -> _OpenAIBackend     (any OpenAI-compatible server)
    """
    backend = (settings.GENERATOR_BACKEND or "fake").strip().lower()

    if backend == "fake":
        return _FakeGenerator()

    if backend == "openai":
        if not settings.OPENAI_BASE_URL:
            raise ValueError(
                "GENERATOR_BACKEND=openai requires OPENAI_BASE_URL "
                "(e.g. http://vllm:8000/v1 or http://localhost:11434/v1)"
            )
        gen: Generator = _OpenAIBackend(
            base_url=settings.OPENAI_BASE_URL,
            model=settings.GENERATOR_MODEL,
            api_key=settings.OPENAI_API_KEY or "",
            temperature=settings.GENERATOR_TEMPERATURE,
            max_tokens=settings.GENERATOR_MAX_TOKENS,
            timeout=settings.GENERATOR_TIMEOUT,
        )
        return gen

    raise ValueError(
        f"Unsupported GENERATOR_BACKEND={backend!r}. "
        f"Expected one of: {', '.join(_SUPPORTED_BACKENDS)}"
    )


def get_embedder(settings: Settings) -> Embedder: ...
def get_retriever(settings: Settings) -> Retriever: ...
def get_reranker(settings: Settings) -> Reranker: ...
