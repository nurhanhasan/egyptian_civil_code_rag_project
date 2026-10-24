"""In-process BGE-M3 retriever — the `RETRIEVER_BACKEND=bge`.

There are two distinct phases:

* Build time (`bge_ingestion.build_local_index`): embed every corpus chunk
  once with BGE-M3 and persist an immutable index directory (`dense.npy`,
  `sparse.jsonl`, `chunks.jsonl`, `manifest.json`).
* Request time: the pipeline embeds only the *question*; `BgeM3Retriever`
  loads the whole corpus index into memory at startup and ranks it with
  Python/NumPy exact brute force — no ANN, no model call inside the retriever,
  and no external vector service.

Fusion is Reciprocal Rank Fusion (RRF) over the dense and sparse channels.
The mandatory `jurisdiction` + `doc_status` filters are applied *before*
scoring, so cross-jurisdiction or repealed leakage is impossible, not merely
unlikely.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ragapp.modules.rag.domain.entities import Chunk, ScoredChunk

DENSE_FILE = "dense.npy"
SPARSE_FILE = "sparse.jsonl"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"

DEFAULT_RRF_K = 60

_CHUNK_FIELDS = tuple(field.name for field in fields(Chunk))
_SPAN_FIELDS = ("norm_span", "raw_span")


@dataclass(frozen=True, slots=True)
class LocalIndexManifest:
    """Provenance of a local index build."""

    embedder_model_id: str
    embedder_revision_id: str
    dim: int
    count: int
    rrf_k: int
    corpus_revision: str
    source_chunks_path: str
    built_at: str


@dataclass(slots=True)
class LocalIndex:
    """Everything needed to answer a query, held in memory per process."""

    manifest: LocalIndexManifest
    chunks: list[Chunk]  # documents/metadata, ranked by index position
    dense: np.ndarray  # [N, dim] float32, L2-normalized
    sparse: list[dict[int, float]]  # per-chunk lexical weights


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {name: getattr(chunk, name) for name in _CHUNK_FIELDS}


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    payload = dict(data)
    for name in _SPAN_FIELDS:
        payload[name] = tuple(payload[name])
    return Chunk(**payload)


def write_local_index(
    out_dir: Path,
    chunks: list[Chunk],
    dense: list[list[float]] | np.ndarray,
    sparse: list[dict[int, float]],
    manifest: LocalIndexManifest,
) -> Path:
    """Persist an immutable local index directory.

    Writes `dense.npy` (dense rows are L2-normalized so a dot product is
    cosine similarity), `sparse.jsonl` (indices + weights per chunk),
    `chunks.jsonl` (full document payload used to rebuild citations), and
    `manifest.json` (embedder/corpus provenance).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(dense, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    np.save(out_dir / DENSE_FILE, matrix)

    with (out_dir / SPARSE_FILE).open("w", encoding="utf-8") as handle:
        for vector in sparse:
            tokens = sorted(vector)
            handle.write(
                json.dumps(
                    {"indices": tokens, "values": [vector[t] for t in tokens]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (out_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False) + "\n")

    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir / MANIFEST_FILE


def load_local_index(index_dir: Path) -> LocalIndex:
    """Load the whole index into memory (called once at process startup).

    Reads chunk payloads, the normalized dense matrix, and the sparse vectors,
    then validates that chunks/dense/sparse all have the same length. All
    subsequent retrieval runs against this in-memory copy — there is no I/O or
    model call per query.
    """
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Local retrieval index not found at {index_dir}. Build it with: "
            "uv run python -m ragapp.scripts.bge_ingest_corpus --out-dir bge_storage"
        )

    manifest = LocalIndexManifest(
        **json.loads(manifest_path.read_text(encoding="utf-8"))
    )

    chunks = [
        _chunk_from_dict(json.loads(line))
        for line in (index_dir / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    dense = np.load(index_dir / DENSE_FILE).astype(np.float32)

    sparse: list[dict[int, float]] = []
    for line in (index_dir / SPARSE_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sparse.append(
            {
                int(token): float(weight)
                for token, weight in zip(record["indices"], record["values"])
            }
        )

    if not len(chunks) == dense.shape[0] == len(sparse):
        raise ValueError(
            "Local index is corrupt: chunks/dense/sparse lengths disagree "
            f"({len(chunks)}/{dense.shape[0]}/{len(sparse)})"
        )

    return LocalIndex(manifest=manifest, chunks=chunks, dense=dense, sparse=sparse)


class BgeM3Retriever:
    """Retriever port backed by a precomputed BGE-M3 local index.

    Hybrid dense + sparse, RRF-fused, always filter-scoped. Each process (and
    each uvicorn worker) holds its own full copy of the corpus in memory; there
    is no lock, no network, and no ANN index. A legal code is small enough for
    exact brute-force scoring, which runs off the event loop.
    """

    def __init__(self, index_dir: str | Path, *, rrf_k: int = DEFAULT_RRF_K) -> None:
        """Load the index directory into memory; `rrf_k` is the fusion constant."""
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        self._rrf_k = rrf_k
        self._index = load_local_index(Path(index_dir))
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

        The arguments are the **query** vectors produced by the pipeline's
        Embedder; this method does not call a model. It validates the mandatory
        filters and query shape, then hands ranking to `_rank` in a worker
        thread so the event loop stays free.

        RAISES:
            ValueError: top_k < 1, an empty mandatory filter value, both
                vectors empty, or a query dense dimension mismatch.
        """
        if not jurisdiction or not doc_status:
            raise ValueError("jurisdiction and doc_status are mandatory filters")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not dense and not sparse:
            raise ValueError(
                "retrieve() requires a non-empty dense and/or sparse vector"
            )

        return await asyncio.to_thread(
            self._rank, list(dense), dict(sparse), top_k, jurisdiction, doc_status
        )

    def _rank(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
        jurisdiction: str,
        doc_status: str,
    ) -> list[ScoredChunk]:
        """Rank the in-memory corpus with Python/NumPy (exact brute force).

        Runs in a worker thread. Steps:

        1. filter corpus indices by `jurisdiction` + `doc_status`;
        2. dense channel: cosine similarity via one matrix multiply
           (`dense[candidates] @ normalized_query`), ranked descending;
        3. sparse channel: dot products between the query token weights and
           each candidate's lexical weights, ranked descending;
        4. RRF fusion: add `1 / (rrf_k + rank + 1)` from each channel into one
           score per chunk;
        5. final sort by fused score (ties by `chunk_id`) and slice to `top_k`.
        """
        index = self._index
        candidates = [
            i
            for i, chunk in enumerate(index.chunks)
            if chunk.jurisdiction == jurisdiction and chunk.doc_status == doc_status
        ]
        if not candidates:
            return []

        fused: dict[int, float] = defaultdict(float)

        if dense:
            query = np.asarray(dense, dtype=np.float32)
            if query.shape[0] != index.dense.shape[1]:
                raise ValueError(
                    f"query dense dimension {query.shape[0]} != index dimension "
                    f"{index.dense.shape[1]}"
                )
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm
            similarities = index.dense[candidates] @ query
            for rank, position in enumerate(np.argsort(-similarities, kind="stable")):
                fused[candidates[int(position)]] += 1.0 / (self._rrf_k + rank + 1)

        if sparse:
            scored: list[tuple[int, float]] = []
            for i in candidates:
                vector = index.sparse[i]
                score = 0.0
                for token, weight in sparse.items():
                    doc_weight = vector.get(token)
                    if doc_weight:
                        score += weight * doc_weight
                if score > 0:
                    scored.append((i, score))
            scored.sort(key=lambda pair: (-pair[1], index.chunks[pair[0]].chunk_id))
            for rank, (i, _) in enumerate(scored):
                fused[i] += 1.0 / (self._rrf_k + rank + 1)

        ranked = sorted(
            fused.items(),
            key=lambda pair: (-pair[1], index.chunks[pair[0]].chunk_id),
        )
        return [
            ScoredChunk(chunk=index.chunks[i], score=score, rank=rank)
            for rank, (i, score) in enumerate(ranked[:top_k])
        ]

    async def close(self) -> None:
        """Idempotent no-op: there is no external resource to release.

        The in-memory index is owned by this instance and freed with it.
        """
        self._closed = True


def build_manifest(
    *,
    embedder_model_id: str,
    embedder_revision_id: str,
    dim: int,
    count: int,
    rrf_k: int,
    corpus_revision: str,
    source_chunks_path: str,
) -> LocalIndexManifest:
    """Construct a manifest with a UTC build timestamp."""
    return LocalIndexManifest(
        embedder_model_id=embedder_model_id,
        embedder_revision_id=embedder_revision_id,
        dim=dim,
        count=count,
        rrf_k=rrf_k,
        corpus_revision=corpus_revision,
        source_chunks_path=source_chunks_path,
        built_at=datetime.now(timezone.utc).isoformat(),
    )
