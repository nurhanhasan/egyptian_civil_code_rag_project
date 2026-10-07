"""
Ports and Adapters pattern for the RAG module.

Port/Interface definition where One *port*, many *adapters*.

`pipeline.py` depends only on the Protocol, never on a concrete vendor/implementation.

`factories.py` is the single selection point that decides WHICH implementation
the pipeline gets, driven purely by config:
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import dataclass


@runtime_checkable
class Generator(Protocol):
    """LLM Generator interface"""

    def generate(self, question: list[dict[str, str]]) -> str:
        """
        Run the model over `question` and return the completion text.

        ARG:
            question: a string tha would be imbedded in a OpenAI-format chat message
                [{"role": "system"|"user"|"assistant", "content": str}, ...]

        RAISES:
            ragapp.exceptions.domain.GeneratorError
        """
        ...

    def close(self) -> None:
        """Release any underlying client/session. Idempotent; a no-op if there is
        nothing to release (the fake)."""
        ...


@dataclass
class EmbeddingResult:
    pass


@runtime_checkable
class Embedder(Protocol):
    """Hybrid by default — returns dense AND sparse + revision_id."""

    async def embed(self, texts: list[str]) -> EmbeddingResult: ...
    def close(self) -> None: ...


@dataclass
class ScoredChunk:
    pass


@runtime_checkable
class Retriever(Protocol):
    """Mandatory filters: jurisdiction + doc_status."""

    async def retrieve(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
        jurisdiction: str,  # REQUIRED - not optional
        doc_status: str,  # REQUIRED - not optional
    ) -> list[ScoredChunk]: ...
    def close(self) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Identity is a real Pareto arm."""

    async def rerank(
        self, query: str, candidates: list[ScoredChunk]
    ) -> list[ScoredChunk]: ...
    def close(self) -> None: ...
