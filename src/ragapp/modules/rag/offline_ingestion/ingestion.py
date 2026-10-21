"""
Corpus ingestion module,
callable from evaluation, offline ingestion, Celery tasks, tests
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient

from ragapp.core.config import Settings
from ragapp.modules.rag.domain.adapters.retrievers import (
    chunk_to_point,
    create_build_collection,
    promote_build,
)
from ragapp.modules.rag.domain.entities import Chunk
from ragapp.modules.rag.factories import get_embedder

# Fixed namespace for deterministic UUID generation from composite IDs
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
BGE_M3_DIM = 1024

DEFAULT_CHUNKS_PATH = Path("data/corpus/processed/chunks.article_aware.jsonl")
DEFAULT_ALIAS = "egyptian-civil-code"


@dataclass
class IngestionResult:
    """Result of an ingestion run."""

    build_id: str
    chunks_indexed: int
    build_duration_ms: float
    corpus_revision: str
    manifest_path: Path


def _compute_corpus_revision(corpus_path: Path) -> str:
    """Compute a content hash of the corpus directory as revision."""
    if not corpus_path.exists():
        return ""
    hashes = []
    for file_path in sorted(corpus_path.rglob("*")):
        if file_path.is_file():
            hashes.append(hashlib.md5(file_path.read_bytes()).hexdigest()[:8])
    return hashlib.md5("".join(hashes).encode()).hexdigest()[:12]


def _section_context(rec: dict) -> str:
    """Build the section breadcrumb used as retrieval/rerank context.

    The article body alone often lacks the wording a user asks with (e.g.
    "بيع ملك الغير" for Article 466). The subsection heading carries it.
    """
    section_path = rec.get("section_path") or []
    titles = [
        str(item.get("title_ar") or "").strip()
        for item in section_path
        if isinstance(item, dict)
    ]
    titles = [title for title in titles if title]
    if titles:
        return " > ".join(titles)
    return rec.get("section_title", {}).get(
        "ar", rec.get("part_name", "Egyptian Civil Code")
    )


def _embedding_text(chunk: Chunk) -> str:
    """Text embedded/matched: section context + normalized article text."""
    if chunk.title and chunk.title not in chunk.norm_text:
        return f"{chunk.title}\n{chunk.norm_text}"
    return chunk.norm_text


def load_chunks(path: Path) -> list[Chunk]:
    """Load and convert JSONL chunks to Chunk entities."""
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: JSON decode error: {e}") from e

            texts_ar = rec.get("texts", {}).get("ar", {}) or {}
            norm_text = texts_ar.get("norm_text", "")
            source_segments = texts_ar.get("source_segments", [])

            if source_segments:
                seg = source_segments[0]
                page = seg.get("page_number")
                norm_span = (
                    seg.get("page_norm_start", 0),
                    seg.get("page_norm_end", len(norm_text)),
                )
                raw_span = (
                    seg.get("page_raw_start", 0),
                    seg.get("page_raw_end", len(norm_text)),
                )
            else:
                page = rec.get("page_numbers", [None])[0]
                norm_span = (0, len(norm_text))
                raw_span = (0, len(norm_text))

            raw_text = norm_text

            chunk = Chunk(
                chunk_id=rec.get("chunk_id", str(uuid.uuid4())),
                doc_id=rec.get("doc_id", "egyptian-civil-law-1948"),
                title=_section_context(rec),
                authority="Egyptian Civil Code",
                jurisdiction=rec.get("jurisdiction", "EG"),
                doc_status=rec.get("doc_status", "in_force"),
                article=rec.get("article"),
                section=rec.get("section_no"),
                page=page,
                norm_text=norm_text,
                raw_text=raw_text,
                norm_span=norm_span,
                raw_span=raw_span,
                source_checksum=rec.get("source_checksum", ""),
                effective_date=rec.get("effective_date"),
                source_url=None,
            )
            chunks.append(chunk)

    return chunks


async def embed_chunks(
    chunks: list[Chunk],
    settings: Settings,
    batch_size: int = 32,
) -> list[tuple[Chunk, list[float], dict[int, float]]]:
    """Embed chunks using the configured Embedder port."""
    embedder = get_embedder(settings)
    texts = [_embedding_text(c) for c in chunks]

    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_chunks = chunks[i : i + batch_size]

        embedding_result = await embedder.embed(batch_texts)
        dense_batch = embedding_result.dense
        sparse_batch = embedding_result.sparse

        for chunk, dense, sparse in zip(batch_chunks, dense_batch, sparse_batch):
            results.append((chunk, dense, sparse))

    await embedder.close()
    return results


def run_ingestion(
    settings: Settings,
    qdrant_url: str | None = None,
    alias: str = DEFAULT_ALIAS,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    batch_size: int = 32,
    upsert_batch: int = 100,
    qdrant_path: Path | str | None = None,
) -> IngestionResult:
    """Run the full ingestion pipeline.

    Args:
        settings: Application settings
        qdrant_url: Qdrant server URL (ignored when `qdrant_path` is set)
        alias: Stable alias name
        chunks_path: Path to chunks JSONL file
        batch_size: Embedding batch size
        upsert_batch: Upsert batch size
        qdrant_path: Build into a Qdrant client local-mode directory instead
            of a server. The directory must not be open by another process.

    Returns:
        IngestionResult with build metadata for MLflow logging
    """
    start_total = time.perf_counter()

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    corpus_revision = _compute_corpus_revision(chunks_path.parent)

    # Load chunks
    chunks = load_chunks(chunks_path)

    # Embed chunks
    embedded = asyncio.run(embed_chunks(chunks, settings, batch_size=batch_size))
    dense_dim = len(embedded[0][1]) if embedded else BGE_M3_DIM

    # Connect to Qdrant (server) or open the local-mode store (no server).
    if qdrant_path is not None:
        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))
    else:
        client = QdrantClient(url=qdrant_url, timeout=60.0)

    try:
        # Create immutable build collection
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        build_id = f"build-{timestamp}-{unique_id}"
        create_build_collection(client, build_id, dense_dim)

        # Upsert points in batches
        for i in range(0, len(embedded), upsert_batch):
            batch = embedded[i : i + upsert_batch]
            points = [
                chunk_to_point(chunk, dense, sparse) for chunk, dense, sparse in batch
            ]
            client.upsert(collection_name=build_id, points=points, wait=True)

        # Promote alias
        promote_build(client, alias, build_id)

        # Verify
        info = client.get_collection(alias)

        build_duration_ms = (time.perf_counter() - start_total) * 1000

        # Create manifest
        manifest = {
            "build_id": build_id,
            "corpus_revision": corpus_revision,
            "corpus_path": str(chunks_path),
            "chunks_indexed": len(chunks),
            "points_in_alias": info.points_count,
            "build_duration_ms": build_duration_ms,
            "qdrant_url": None if qdrant_path is not None else qdrant_url,
            "qdrant_path": str(qdrant_path) if qdrant_path is not None else None,
            "alias": alias,
            "embedder_model_id": settings.BGE_M3_MODEL_PATH,
            "embedder_revision_id": settings.EMBEDDER_MODEL_REVISION_ID,
        }
        manifest_path = Path(f"manifest_{build_id}.json")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

        return IngestionResult(
            build_id=build_id,
            chunks_indexed=len(chunks),
            build_duration_ms=build_duration_ms,
            corpus_revision=corpus_revision,
            manifest_path=manifest_path,
        )

    finally:
        client.close()
