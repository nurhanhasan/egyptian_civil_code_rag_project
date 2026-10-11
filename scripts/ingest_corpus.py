#!/usr/bin/env python3
"""Full corpus ingestion pipeline for Egyptian Civil Code.

Reads processed chunks from data/corpus/processed/chunks.article_aware.jsonl,
embeds with configured Embedder (BGE-M3 by default), creates immutable build
collection, upserts, and promotes the stable alias.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient

from ragapp.core.config import get_settings
from ragapp.modules.rag.domain.adapters.retrievers import (
    chunk_to_point,
    create_build_collection,
    promote_build,
)
from ragapp.modules.rag.domain.entities import Chunk
from ragapp.modules.rag.factories import get_embedder

CHUNKS_PATH = (
    Path(__file__).parent.parent / "data/corpus/processed/chunks.article_aware.jsonl"
)
DEFAULT_ALIAS = "egyptian-civil-code"
DEFAULT_QDRANT_URL = "http://localhost:6333"
BGE_M3_DIM = 1024

# Fixed namespace for deterministic UUID generation from composite IDs
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def load_chunks(path: Path) -> list[Chunk]:
    """Load and convert JSONL chunks to Chunk entities."""
    chunks = []
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: line {line_num}: JSON decode error: {e}", file=sys.stderr)
                continue

            # Map JSONL fields to Chunk entity
            # The source JSONL has: chunk_id, texts.ar.norm_text, texts.ar.source_segments, etc.  # noqa: E501
            # We need: chunk_id, doc_id, title, authority, jurisdiction, doc_status,  # noqa: E501
            # article, section, page, norm_text, raw_text, norm_span, raw_span,  # noqa: E501
            # source_checksum, effective_date, source_url

            texts_ar = rec.get("texts", {}).get("ar", {}) or {}
            norm_text = texts_ar.get("norm_text", "")
            source_segments = texts_ar.get("source_segments", [])

            # Use the first segment for span info, or defaults
            if source_segments:
                seg = source_segments[0]
                page = seg.get("page_number")
                # Use norm_start/norm_end for norm_span
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

            # raw_text = norm_text for now (the norm_text is already the Arabic text)
            # In the full pipeline, there'd be an offset map; here they're identical  # noqa: E501
            raw_text = norm_text

            chunk = Chunk(
                chunk_id=rec.get("chunk_id", str(uuid.uuid4())),
                doc_id=rec.get("doc_id", "egyptian-civil-law-1948"),
                title=rec.get("section_title", {}).get(
                    "ar", rec.get("part_name", "Egyptian Civil Code")
                ),
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
    chunks: list[Chunk], batch_size: int = 32
) -> list[tuple[Chunk, list[float], dict[int, float]]]:
    """Embed chunks using the configured Embedder port."""
    settings = get_settings()
    embedder = get_embedder(settings)
    print(f"Using embedder: {embedder.__class__.__name__}")
    print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

    results = []
    texts = [c.norm_text for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_chunks = chunks[i : i + batch_size]

        # Embedder port returns EmbeddingResult with dense, sparse, revision_id
        embedding_result = await embedder.embed(batch_texts)

        dense_batch = embedding_result.dense
        sparse_batch = embedding_result.sparse

        for j, (chunk, dense, sparse) in enumerate(
            zip(batch_chunks, dense_batch, sparse_batch)
        ):
            results.append((chunk, dense, sparse))

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    embedder.close()
    print("Embedding complete.")
    return results


def run_ingestion(
    qdrant_url: str = DEFAULT_QDRANT_URL,
    alias: str = DEFAULT_ALIAS,
    batch_size: int = 32,
    upsert_batch: int = 100,
) -> str:
    """Run the full ingestion pipeline."""
    print("=== Egyptian Civil Code Corpus Ingestion ===")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Target alias: {alias}")
    print(f"Chunks file: {CHUNKS_PATH}")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")

    # Load chunks
    print(f"\n1. Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"   Loaded {len(chunks)} chunks")

    # Embed chunks
    print("\n2. Embedding chunks with configured Embedder...")
    start = time.time()
    embedded = asyncio.run(embed_chunks(chunks, batch_size=batch_size))
    print(f"   Embedding took {time.time() - start:.1f}s")

    # Connect to Qdrant
    print(f"\n3. Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, timeout=60.0)

    # Create immutable build collection
    build_id = f"build-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"  # noqa: E501
    print(f"\n4. Creating immutable build collection: {build_id}")
    create_build_collection(client, build_id, BGE_M3_DIM)

    # Upsert points in batches
    print(f"\n5. Upserting {len(embedded)} points in batches of {upsert_batch}...")
    start = time.time()
    for i in range(0, len(embedded), upsert_batch):
        batch = embedded[i : i + upsert_batch]
        points = [
            chunk_to_point(chunk, dense, sparse) for chunk, dense, sparse in batch
        ]
        client.upsert(collection_name=build_id, points=points, wait=True)
        if (i // upsert_batch + 1) % 5 == 0:
            print(
                f"   Upserted {min(i + upsert_batch, len(embedded))}/{len(embedded)} points"  # noqa: E501
            )
    print(f"   Upsert took {time.time() - start:.1f}s")

    # Promote alias
    print(f"\n6. Promoting alias '{alias}' -> '{build_id}'...")
    promote_build(client, alias, build_id)
    print("   Alias promoted successfully!")

    # Verify
    print("\n7. Verifying alias points to build...")
    info = client.get_collection(alias)
    print(f"   Collection '{alias}' has {info.points_count} points")

    client.close()
    print("\n=== Ingestion complete! ===")
    return build_id


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Egyptian Civil Code corpus into Qdrant"
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="Stable alias name")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
    )
    parser.add_argument(
        "--upsert-batch", type=int, default=100, help="Upsert batch size"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run a test query after ingestion"
    )
    args = parser.parse_args()

    build_id = run_ingestion(
        qdrant_url=args.qdrant_url,
        alias=args.alias,
        batch_size=args.batch_size,
        upsert_batch=args.upsert_batch,
    )

    if args.verify:
        print("\n=== Running verification query ===")
        client = QdrantClient(url=args.qdrant_url, timeout=30.0)
        # Quick test: query for article 1
        from ragapp.modules.rag.domain.adapters.retrievers import QdrantRetriever

        retriever = QdrantRetriever(client, args.alias)
        import asyncio

        async def test():
            results = await retriever.retrieve(
                dense=[0.0] * BGE_M3_DIM,  # dummy
                sparse={},
                top_k=3,
                jurisdiction="EG",
                doc_status="in_force",
            )
            return results

        results = asyncio.run(test())
        print(f"Test query returned {len(results)} results")
        for r in results:
            print(f"  - {r.chunk.article}: {r.chunk.raw_text[:80]}...")
        client.close()

    print(f"\nBuild ID: {build_id}")
    print(f"Alias: {args.alias} -> {build_id}")


if __name__ == "__main__":
    main()
