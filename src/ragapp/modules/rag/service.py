from __future__ import annotations

import logging
from typing import Any

from langfuse import LangfuseSpan, get_client

from ragapp.modules.rag.domain.entities import ScoredChunk
from ragapp.modules.rag.domain.ports import Embedder, Generator, Reranker, Retriever
from ragapp.modules.rag.schema import AbstentionResponse, AskResponse, Citation

logger = logging.getLogger(__name__)

# Abstention thresholds (M1 #33, SPEC.md §1.2)
MIN_RERANK_SCORE = 0.15
MAX_CITATIONS = 5
RETRIEVAL_TOP_K = 20


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

    def _build_context(self, chunks: list[ScoredChunk]) -> str:
        """Build context string from retrieved chunks for the prompt."""
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks[:MAX_CITATIONS], 1):
            article = chunk.chunk.article or "unknown"
            title = chunk.chunk.title
            raw_text = chunk.chunk.raw_text
            context_parts.append(
                f"[Source {i} - Article {article}]\n{title}\n{raw_text}"
            )

        return "\n\n".join(context_parts)

    def _build_messages(self, question: str, context: str) -> list[dict[str, str]]:
        """Assemble the chat messages for the generator with context."""
        user_content = f"""
        Question: {question}

        Context from Egyptian Civil Code: {context}
        """

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _build_citations(self, chunks: list[ScoredChunk]) -> list[Citation]:
        """Build citation objects from reranked chunks.

        Only raw_text is quoted. Sorted by score desc, capped at MAX_CITATIONS.
        """
        citations = []
        for chunk in chunks[:MAX_CITATIONS]:
            citations.append(
                Citation(
                    article=chunk.chunk.article or "unknown",
                    raw_text=chunk.chunk.raw_text,
                    source_checksum=chunk.chunk.source_checksum,
                    effective_date=chunk.chunk.effective_date,
                    doc_status=chunk.chunk.doc_status,
                    jurisdiction=chunk.chunk.jurisdiction,
                    score=chunk.score,
                    rank=chunk.rank,
                )
            )
        return citations

    def _check_abstention(
        self, reranked: list[ScoredChunk], answer: str
    ) -> tuple[bool, str | None]:
        """Check abstention conditions (SPEC.md §1.2).

        Returns (should_abstain, reason).
        """
        # 1. No retrievable chunks after mandatory filters
        if not reranked:
            return True, "no_relevant_documents"

        # 2. Top reranked chunk score below threshold
        if reranked[0].score < MIN_RERANK_SCORE:
            return True, f"low_confidence_score_{reranked[0].score:.3f}"

        # 3. Generator returns empty/refusal
        if not answer or answer.strip() == "":
            return True, "empty_generation"

        # 4. Citation resolution fails (staleness guards) - checked in _build_citations
        #    If any citation missing required fields, abstain
        for chunk in reranked[:MAX_CITATIONS]:
            if not chunk.chunk.source_checksum:
                return True, "missing_source_checksum"
            if not chunk.chunk.doc_status:
                return True, "missing_doc_status"

        return False, None

    def _instrument(
        self, span: LangfuseSpan, as_type: str, name: str, input: Any, output: Any
    ) -> None:
        child = span.start_observation(name=name, as_type=as_type)
        child.update(input=input)
        child.update(output=output)
        child.end()

    # @observe()
    async def run_pipeline(
        self, question: str
    ) -> tuple[AskResponse | AbstentionResponse, str]:
        """Answer `question` via the configured pipeline.

        Returns structured response with citations or abstention.
        """

        # TODO: Initialize once at startup
        try:
            langfuse_client = get_client()
            langfuse_client.auth_check()
        except Exception as e:
            logger.warning("Langfuse authentication check failed: %s", e)

        with langfuse_client.start_as_current_observation(
            name="main-span", input=question
        ) as span:
            with span.start_as_current_observation(
                name="embedding-question", as_type="embedding", input=question
            ) as embedding_span:
                question_embeddings = await self._embedder.embed([question])
                embedding_span.update(output=question_embeddings)

            with span.start_as_current_observation(
                name="retrieving",
                as_type="retriever",
                input=question_embeddings,
            ) as retrieval_span:
                retrieved_chunks = await self._retriever.retrieve(
                    dense=question_embeddings.dense[0],
                    sparse=question_embeddings.sparse[0],
                    top_k=RETRIEVAL_TOP_K,
                    jurisdiction="EG",
                    doc_status="in_force",
                )
                retrieval_span.update(
                    output=[chunk.chunk.raw_text for chunk in retrieved_chunks]
                )

            with span.start_as_current_observation(
                name="reranker", as_type="evaluator", input=retrieved_chunks
            ) as reranker_span:
                reranked = await self._reranker.rerank(question, retrieved_chunks)
                reranker_span.update(output=reranked)

            # Check abstention before generation
            if not reranked or not reranked[0].chunk:
                return AbstentionResponse(reason="no_relevant_documents"), span.trace_id

            with span.start_as_current_observation(
                name="generate", as_type="generation", input=reranked
            ) as generation_span:
                context = self._build_context(reranked)
                messages = self._build_messages(question, context)

                answer = await self._generator.generate(messages)

                generation_span.update(output=answer)

            span.update(output=answer)

            with span.start_as_current_observation(
                name="citation", as_type="span", input=reranked
            ) as citation_span:
                # Post-generation abstention checks
                should_abstain, reason = self._check_abstention(reranked, answer)
                if should_abstain:
                    return AbstentionResponse(reason=reason), span.trace_id

                # Build citations from reranked chunks
                citations = self._build_citations(reranked)
                citation_span.update(
                    output=[citation.article for citation in citations]
                )

            langfuse_client.flush()

        return AskResponse(
            answer=answer,
            sources=[citation.article for citation in citations],
            trace_id=span.trace_id,
        ), span.trace_id

    async def close(self) -> None:
        await self._generator.close()
        await self._embedder.close()
        await self._retriever.close()
        await self._reranker.close()
