import asyncio
import os
import re

import bentoml
from FlagEmbedding import FlagReranker

from ragapp.modules.rag.domain.entities import Chunk, ScoredChunk

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL_REVISION_ID = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

SECTION_OVERLAP_WEIGHT = 1.0
SECTION_BIGRAM_WEIGHT = 1.0

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "ما",
        "ماذا",
        "هي",
        "هو",
        "في",
        "من",
        "على",
        "علي",
        "عن",
        "الى",
        "إلى",
        "و",
        "او",
        "أو",
        "متى",
        "متي",
        "كيف",
        "هل",
        "الذي",
        "التي",
        "بين",
        "الفرق",
        "حكم",
        "احكام",
        "أحكام",
        "شروط",
        "صحة",
        "صحه",
        "اثار",
        "آثار",
        "ماده",
        "مادة",
    }
)


def _arabic_tokens(text: str) -> list[str]:
    """Light Arabic normalization for section-heading matching."""
    normalized = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    normalized = _DIACRITICS.sub("", normalized)
    normalized = _NON_WORD.sub(" ", normalized)

    tokens: list[str] = []
    for token in normalized.casefold().split():
        if token.startswith("ال") and len(token) > 3:
            token = token[2:]
        if len(token) > 1 and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def _section_heading(chunk: Chunk) -> str:
    return chunk.title.rsplit(" > ", 1)[-1] if chunk.title else ""


def _section_boost(question: str, chunk: Chunk) -> float:
    """Heading/phrase overlap bonus for the cross-encoder score.

    The general cross-encoder often ranks articles from a semantically nearby
    section above the exact subsection the question names. Exact heading token
    overlap (e.g. "ملك الغير" in "بيع ملك الغير") is a high-precision signal.
    """
    question_tokens = set(_arabic_tokens(question))
    heading_tokens = _arabic_tokens(_section_heading(chunk))
    if not question_tokens or not heading_tokens:
        return 0.0

    overlap = len(question_tokens & set(heading_tokens))
    bigrams = sum(
        1
        for left, right in zip(heading_tokens, heading_tokens[1:])
        if left in question_tokens and right in question_tokens
    )
    return SECTION_OVERLAP_WEIGHT * overlap + SECTION_BIGRAM_WEIGHT * bigrams


def _rerank_text(chunk: Chunk) -> str:
    """Most specific section heading + raw article text.

    The retrieval index uses the full breadcrumb, but the cross-encoder is more
    accurate with the concise subsection heading (e.g. "بيع ملك الغير").
    raw_text remains the only text quoted in citations.
    """
    heading = _section_heading(chunk)
    if heading and heading not in chunk.raw_text:
        return f"{heading}\n{chunk.raw_text}"
    return chunk.raw_text


def _rank_scored(
    candidates: list[ScoredChunk],
    scores: list[float],
    *,
    question: str = "",
) -> list[ScoredChunk]:
    """Combine cross-encoder scores with section-heading evidence, then sort."""
    scored = [
        ScoredChunk(
            chunk=candidate.chunk,
            score=float(scores[i]) + _section_boost(question, candidate.chunk),
            rank=0,
        )
        for i, candidate in enumerate(candidates)
    ]
    ranked = sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))
    return [
        ScoredChunk(chunk=item.chunk, score=item.score, rank=rank)
        for rank, item in enumerate(ranked)
    ]


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
