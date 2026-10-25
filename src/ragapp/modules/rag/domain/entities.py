from dataclasses import dataclass

### Embedder


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Output of Embedder.port — hybrid by default."""

    dense: list[list[float]]  # [batch, 1024] for BGE-M3
    sparse: list[dict[int, float]]  # Lexical-weighted sparse vectors (Qdrant format)
    revision_id: str  # Pinned model revision (e.g., "bge-m3@abc123def")


### Reranker


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk from the corpus — article-aware."""

    chunk_id: str  # UUID
    doc_id: str  # Source document ID
    title: str  # Document title
    authority: str  # e.g., "Egyptian Civil Code"
    jurisdiction: str  # e.g., "EG" — mandatory filter
    doc_status: str  # e.g., "active" | "repealed" | "amended" — mandatory filter
    article: str | None  # Article number (e.g., "147")
    section: str | None  # Section within article
    page: int | None  # Page number
    norm_text: str  # Normalized text (embedded/matched)
    raw_text: str  # Raw text (quoted in citations)
    norm_span: tuple[int, int]  # Span in norm_text
    raw_span: tuple[int, int]  # Span in raw_text (via offset map)
    source_checksum: str  # SHA256 of raw_text — citation staleness guard
    effective_date: str | None  # ISO date — staleness guard
    source_url: str | None  # Verifiability


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """Retriever/Reranker output — chunk with score.

    `score` orders the candidates for the stage that produced them (dense,
    sparse, RRF, or cross-encoder logit plus heading bonus). `confidence` is a
    probability in (0, 1) that only a cross-encoder can supply; the abstention
    gate reads it, never `score`, because a boosted logit is not comparable to a
    probability threshold.
    """

    chunk: Chunk
    score: float  # Retrieval score (dense/sparse/RRF) or rerank score
    rank: int  # 0-based rank after this stage
    confidence: float | None = None  # Cross-encoder P(relevant); None before rerank
