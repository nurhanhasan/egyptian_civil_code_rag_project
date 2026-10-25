"""Reranker scoring math — no model, no torch, no BentoML.

Kept separate from `rerankers.py` so the ranking and confidence rules are
unit-testable without importing FlagEmbedding. `rerankers.py` owns the model;
this module owns what the numbers mean.

Two numbers come out of `_rank_scored`:

* `score` — the ranking key: raw cross-encoder logit + section-heading bonus.
* `confidence` — `sigmoid(logit)`, the boost-free probability that the chunk
  answers the question. Only this one may be compared to a threshold.
"""

from __future__ import annotations

import math
import re

from ragapp.modules.rag.domain.entities import Chunk, ScoredChunk

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


def _sigmoid(value: float) -> float:
    """Logistic function, stable for large |value|."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


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
    scores: list[float] | float,
    *,
    question: str = "",
) -> list[ScoredChunk]:
    """Combine cross-encoder scores with section-heading evidence, then sort.

    `score` stays the ranking key (logit + heading bonus). `confidence` is the
    boost-free probability the cross-encoder assigns to the pair, and it is the
    only number the abstention gate may read: adding a heading bonus to a logit
    destroys its comparability with any probability threshold.
    """
    if not isinstance(scores, (list, tuple)):
        # FlagReranker returns a bare float when handed a single pair.
        scores = [float(scores)]

    scored = [
        ScoredChunk(
            chunk=candidate.chunk,
            score=float(scores[i]) + _section_boost(question, candidate.chunk),
            rank=0,
            confidence=_sigmoid(float(scores[i])),
        )
        for i, candidate in enumerate(candidates)
    ]
    ranked = sorted(scored, key=lambda item: (-item.score, item.chunk.chunk_id))
    return [
        ScoredChunk(
            chunk=item.chunk,
            score=item.score,
            rank=rank,
            confidence=item.confidence,
        )
        for rank, item in enumerate(ranked)
    ]
