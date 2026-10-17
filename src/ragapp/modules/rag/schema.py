from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    Citation for a retrieved chunk.

    Only `raw_text` is quoted in responses.
    """

    article: str
    raw_text: str
    source_checksum: str
    effective_date: str | None = None
    doc_status: str
    jurisdiction: str = "EG"
    score: float
    rank: int


class AskRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=2000)]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class AbstentionResponse(BaseModel):
    """
    Abstention response.

    Four refusal conditions:
    1. No retrievable chunks after mandatory filters
    2. Top reranked chunk score below threshold
    3. Generator returns empty/refusal
    4. Citation resolution fails (staleness guards)
    """

    abstained: Literal[True] = True
    reason: str
    answer: None = None
    sources: list[Citation] = []


# Union for OpenAPI schema
AskResult = Annotated[
    AskResponse | AbstentionResponse, Field(discriminator="abstained")
]
