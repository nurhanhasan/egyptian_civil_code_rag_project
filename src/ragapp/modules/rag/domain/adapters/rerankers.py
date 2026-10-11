import asyncio
import os

import bentoml
from FlagEmbedding import FlagLLMReranker

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
            ScoredChunk(chunk=c.chunk, score=0.8 - i * 0.05, rank=i)
            for i, c in enumerate(candidates)
        ]

    def close(self) -> None:
        pass


@bentoml.service()
class BentoMLBGEReranker:
    """Reranker implementation using BGE reranker via BentoML."""

    def __init__(self):
        # @bentoml.service() services are instantiated by BentoML runtime,
        # not your factory. BentoML calls __init__() with no args.
        self.model_path = os.getenv("BGE_RERANKER_MODEL_PATH", RERANKER_MODEL_ID)
        self.model = FlagLLMReranker(self.model_path, use_fp16=True)
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

        pairs = [[question, c.chunk.raw_text] for c in candidates]

        scores = await asyncio.to_thread(self.model.compute_score, pairs)

        return [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(scores[i]),
                rank=i,
            )
            for i, candidate in enumerate(candidates)
        ]

    def close(self) -> None:
        self.model = None
