import asyncio
import os

import bentoml
from FlagEmbedding import FlagReranker

from ragapp.modules.rag.domain.adapters.rerank_scoring import (
    _rank_scored,
    _rerank_text,
)
from ragapp.modules.rag.domain.entities import ScoredChunk

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION_ID = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class _FakeReranker:
    """Deterministic fake reranker for tests."""

    async def rerank(
        self, question: str, candidates: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        return [
            ScoredChunk(
                chunk=c.chunk,
                score=0.8 - i * 0.05,
                rank=i,
                confidence=max(0.0, min(1.0, 0.8 - i * 0.05)),
            )
            for i, c in enumerate(candidates)
        ]

    async def close(self) -> None:
        pass


@bentoml.service()
class BentoMLBGEReranker:
    """Reranker implementation using BGE reranker via BentoML."""

    def __init__(self, device: str | None = None):
        # @bentoml.service() wraps the class: zero-arg `Service()` yields a
        # default instance for the BentoML runtime path, while `Service.inner`
        # is the raw class the factory uses to inject config (e.g. device).
        self.model_path = os.getenv("BGE_RERANKER_MODEL_PATH", RERANKER_MODEL_ID)
        self.model = FlagReranker(self.model_path, use_fp16=True, devices=device)
        self.revision_id = os.getenv(
            "RERANKER_MODEL_REVISION_ID", RERANKER_MODEL_REVISION_ID
        )

    @bentoml.api(batchable=True, max_batch_size=32)
    async def rerank(
        self, question: str, candidates: list[ScoredChunk]
    ) -> list[ScoredChunk]:
        """Rerank candidates using the cross-encoder model.

        Runs the blocking compute_score in a thread pool to avoid blocking
        the event loop (port requires async).
        """
        if not candidates:
            return []

        pairs = [[question, _rerank_text(c.chunk)] for c in candidates]

        scores = await asyncio.to_thread(self.model.compute_score, pairs)

        return _rank_scored(candidates, scores, question=question)

    async def close(self) -> None:
        self.model = None
