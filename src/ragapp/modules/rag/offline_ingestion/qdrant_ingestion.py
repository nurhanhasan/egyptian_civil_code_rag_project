"""
Corpus ingestion module,
callable from evaluation, offline ingestion, Celery tasks, tests
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient

from ragapp.core.config import Settings
from ragapp.modules.rag.domain.adapters.qdrant_retriever import (
    chunk_to_point,
    create_build_collection,
    promote_build,
)
from ragapp.modules.rag.offline_ingestion.common import (
    compute_corpus_revision,
    embed_chunks,
    load_chunks,
)

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


def run_ingestion(
    settings: Settings,
    qdrant_url: str | None = None,
    alias: str = DEFAULT_ALIAS,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    batch_size: int = 32,
    upsert_batch: int = 100,
    qdrant_path: Path | str | None = None,
) -> IngestionResult:
    """Run the full offline ingestion pipeline into a Qdrant store.

    Steps: load chunks -> embed dense + sparse -> create an immutable
    `build-*` collection (named vectors + filter indexes) -> batch upsert ->
    promote the stable alias -> verify -> write a build manifest. The client is
    either a Qdrant server (`qdrant_url`) or the in-process local store
    (`qdrant_path`); no retrieval happens here.

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

    corpus_revision = compute_corpus_revision(chunks_path)

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
