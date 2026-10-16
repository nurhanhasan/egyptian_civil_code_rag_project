"""Qdrant adapter for the Retriever port.

The retriever reads the stable ALIAS, never a collection name;
promotion/rollback is an alias change performed offline.

Retrieval is never unfiltered — `jurisdiction` + `doc_status` are mandatory
on every query (cross-jurisdiction leakage is a wrong answer, not a ranking
imperfection).

Hybrid is the default — a named dense vector and a named sparse vector
combined with Reciprocal Rank Fusion. The RRF `k`parameter is configuration,
never hardcoded.

`create_build_collection` / `chunk_to_point` / `promote_build` are OFFLINE
(ingestion) operations: immutable index builds and alias promotion.
`QdrantRetriever` is request-path only.
"""

from __future__ import annotations

import asyncio
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qm
from qdrant_client.http.models import (
    CreateAliasOperation,
    NearestQuery,
    RenameAliasOperation,
    Rrf,
    RrfQuery,
    SparseVector,
)

from ragapp.modules.rag.domain.entities import Chunk, ScoredChunk

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DEFAULT_RRF_K = 60

#: Payload fields with an index. Every query filters on both.
MANDATORY_FILTER_FIELDS = ("jurisdiction", "doc_status")

# Fixed namespace for deterministic UUID generation from composite IDs
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _to_point_id(chunk_id: str) -> uuid.UUID:
    """Convert any chunk_id to a deterministic UUID for Qdrant point ID.

    If chunk_id is already a valid UUID, use it. Otherwise, generate a
    deterministic UUID5 from the string using a fixed namespace.
    """
    try:
        return uuid.UUID(chunk_id)
    except ValueError:
        return uuid.uuid5(NAMESPACE, chunk_id)


def _chunk_id_from_point(point_id: uuid.UUID, payload: dict | None) -> str:
    """Recover the original chunk_id from point ID or payload."""
    if payload and "chunk_id" in payload:
        return payload["chunk_id"]
    return str(point_id)


def _mandatory_filter(jurisdiction: str, doc_status: str) -> qm.Filter:
    """Both conditions, always — an unfiltered query cannot be built here."""
    if not jurisdiction or not doc_status:
        raise ValueError("jurisdiction and doc_status are mandatory filters")
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="jurisdiction", match=qm.MatchValue(value=jurisdiction)
            ),
            qm.FieldCondition(key="doc_status", match=qm.MatchValue(value=doc_status)),
        ]
    )


def _to_sparse_vector(sparse: dict[int, float]) -> SparseVector:
    """Port shape {token_id: weight} -> Qdrant {indices, values}.

    Sorted so the wire representation is byte-stable regardless of the
    embedder's dict ordering (the dot product itself is order-insensitive).
    """
    token_ids = sorted(sparse)
    return SparseVector(indices=token_ids, values=[sparse[t] for t in token_ids])


def create_build_collection(client: QdrantClient, build_id: str, dense_dim: int) -> str:
    """Create a NEW immutable index build. Returns the collection name.

    Ingestion never mutates a live collection: the build is a fresh
    collection named by its immutable `build_id`, with the named dense +
    sparse vectors and payload indexes on both mandatory filter
    fields. The request path only ever sees the alias.
    """
    client.create_collection(
        collection_name=build_id,
        vectors_config={
            DENSE_VECTOR: qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE_VECTOR: qm.SparseVectorParams()},
    )
    for field in MANDATORY_FILTER_FIELDS:
        client.create_payload_index(
            build_id, field, field_schema=qm.PayloadSchemaType.KEYWORD
        )
    return build_id


def chunk_to_point(
    chunk: Chunk, dense: list[float], sparse: dict[int, float]
) -> qm.PointStruct:
    """Chunk + its vectors -> a Qdrant point (ingestion-side schema owner).

    The payload carries every `Chunk` field so the request path can rebuild
    citations verbatim: `raw_text` is the only text quoted in citations,
    `source_checksum` + `effective_date` guard staleness.
    """
    return qm.PointStruct(
        id=_to_point_id(chunk.chunk_id),
        vector={
            DENSE_VECTOR: list(dense),
            SPARSE_VECTOR: _to_sparse_vector(sparse),
        },
        payload={
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "authority": chunk.authority,
            "jurisdiction": chunk.jurisdiction,
            "doc_status": chunk.doc_status,
            "article": chunk.article,
            "section": chunk.section,
            "page": chunk.page,
            "norm_text": chunk.norm_text,
            "raw_text": chunk.raw_text,
            "norm_span": list(chunk.norm_span),
            "raw_span": list(chunk.raw_span),
            "source_checksum": chunk.source_checksum,
            "effective_date": chunk.effective_date,
            "source_url": chunk.source_url,
        },
    )


def chunk_from_point(point_id: object, payload: dict | None) -> Chunk:
    """Inverse of `chunk_to_point`. Raises on a malformed build."""
    if payload is None:
        raise ValueError(f"point {point_id} has no payload — malformed index build")
    return Chunk(
        chunk_id=_chunk_id_from_point(point_id, payload),
        doc_id=payload["doc_id"],
        title=payload["title"],
        authority=payload["authority"],
        jurisdiction=payload["jurisdiction"],
        doc_status=payload["doc_status"],
        article=payload["article"],
        section=payload["section"],
        page=payload["page"],
        norm_text=payload["norm_text"],
        raw_text=payload["raw_text"],
        norm_span=(int(payload["norm_span"][0]), int(payload["norm_span"][1])),
        raw_span=(int(payload["raw_span"][0]), int(payload["raw_span"][1])),
        source_checksum=payload["source_checksum"],
        effective_date=payload["effective_date"],
        source_url=payload["source_url"],
    )


def promote_build(client: QdrantClient, alias: str, build_id: str) -> None:
    """Point the stable alias at `build_id` (promotion).

    Rollback is the same call with the previous build id. One batch of
    two alias operations: a temp alias on the new build, renamed
    over the live alias.
    """
    temp_alias = f"{alias}__next"
    client.update_collection_aliases(
        [
            CreateAliasOperation(
                create_alias=qm.CreateAlias(
                    collection_name=build_id, alias_name=temp_alias
                )
            ),
            RenameAliasOperation(
                rename_alias=qm.RenameAlias(
                    old_alias_name=temp_alias, new_alias_name=alias
                )
            ),
        ]
    )


class QdrantRetriever:
    """Retriever port implementation backed by Qdrant.

    Reads the stable alias only; promotion/rollback is an offline alias
    change, never an image rebuild or collection mutation.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_alias: str,
        *,
        dense_vector_name: str = DENSE_VECTOR,
        sparse_vector_name: str = SPARSE_VECTOR,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not collection_alias:
            raise ValueError(
                "collection_alias is required — the request path reads the "
                "stable alias, never a collection name"
            )
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        self._client = client
        self._alias = collection_alias
        self._dense_vector = dense_vector_name
        self._sparse_vector = sparse_vector_name
        self._rrf_k = rrf_k
        self._closed = False

    async def retrieve(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
        jurisdiction: str,
        doc_status: str,
    ) -> list[ScoredChunk]:
        """Hybrid dense + sparse retrieval, RRF-fused, always filter-scoped.

        Arguments mirror the Retriever port (ports.py): single-query
        vectors, `jurisdiction` + `doc_status` as required positional
        parameters so unfiltered retrieval is a type error.

        RAISES:
            ValueError: top_k < 1, an empty mandatory filter value, or both
                vectors empty.
        """
        flt = _mandatory_filter(jurisdiction, doc_status)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        prefetches = []
        if dense:
            prefetches.append(
                qm.Prefetch(
                    query=NearestQuery(nearest=list(dense)),
                    using=self._dense_vector,
                    filter=flt,
                    limit=top_k,
                )
            )
        if sparse:
            prefetches.append(
                qm.Prefetch(
                    query=NearestQuery(nearest=_to_sparse_vector(sparse)),
                    using=self._sparse_vector,
                    filter=flt,
                    limit=top_k,
                )
            )
        if not prefetches:
            raise ValueError(
                "retrieve() requires a non-empty dense and/or sparse vector"
            )

        response = await asyncio.to_thread(
            self._client.query_points,
            collection_name=self._alias,
            query=RrfQuery(rrf=Rrf(k=self._rrf_k)),
            prefetch=prefetches,
            query_filter=flt,
            limit=top_k,
        )
        return [
            ScoredChunk(
                chunk=chunk_from_point(point.id, point.payload),
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points)
        ]

    async def close(self) -> None:
        """Release the client. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._client.close()


class _FakeRetriever:
    """Deterministic fake retriever for tests.

    Returns canned chunks matching the filter (jurisdiction + doc_status).
    """

    def __init__(self):
        self._chunks = [
            Chunk(
                chunk_id="fake-chunk-1",
                doc_id="doc-1",
                title="المادة 147",
                authority="Egyptian Civil Code",
                jurisdiction="EG",
                doc_status="in_force",
                article="147",
                section=None,
                page=1,
                norm_text="العقد الرضا واجبة",
                raw_text="العقد الرضا واجبة",
                norm_span=(0, 16),
                raw_span=(0, 16),
                source_checksum="sha256:doc-1",
                effective_date="2020-01-01",
                source_url=None,
            ),
            Chunk(
                chunk_id="fake-chunk-2",
                doc_id="doc-2",
                title="المادة 148",
                authority="Egyptian Civil Code",
                jurisdiction="EG",
                doc_status="in_force",
                article="148",
                section=None,
                page=1,
                norm_text="الفسخ بالشرط الفاسخ",
                raw_text="الفسخ بالشرط الفاسخ",
                norm_span=(0, 18),
                raw_span=(0, 18),
                source_checksum="sha256:doc-2",
                effective_date="2020-01-01",
                source_url=None,
            ),
        ]

    async def retrieve(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
        jurisdiction: str,
        doc_status: str,
    ) -> list[ScoredChunk]:
        if jurisdiction != "EG" or doc_status != "in_force":
            return []
        return [
            ScoredChunk(chunk=chunk, score=0.9 - i * 0.1, rank=i)
            for i, chunk in enumerate(self._chunks[:top_k])
        ]

    async def close(self) -> None:
        pass
