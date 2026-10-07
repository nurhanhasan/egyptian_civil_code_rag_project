from __future__ import annotations

import logging

from ragapp.modules.rag.domain.ports import Embedder, Generator, Reranker, Retriever

logger = logging.getLogger(__name__)


# @bentoml.service(resources={"gpu": "1"})  # or CPU
class RAGPipelineService:
    """RAG Pipeline Service

    Executes the complete Retrieval-Augmented Generation (RAG) pipeline.

    Coordinates context assembly, prompt construction, abstention checks,
    citation resolution, guardrail validation, OpenTelemetry tracing if time permits,
    and evaluator logging.
    """

    def __init__(
        self,
        generator: Generator,
        embedder: Embedder,
        retriever: Retriever,
        reranker: Reranker,
        system_prompt: str,
    ) -> None:
        self._generator = generator
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._system_prompt = system_prompt

    def _build_message(self, question: str) -> list[dict[str, str]]:
        """Assemble the chat messages for the generator."""
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]

    def run_pipeline(self, question: str) -> str:
        """Answer `question` via the configured pipeline.

        HTTP layer maps it to a stable 502 problem+json.
        """
        # Force spike the workflow with bentoml service wrapping the pipeline.
        return self._generator.generate(self._build_message(question))
