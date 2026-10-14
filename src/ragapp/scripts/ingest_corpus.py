#!/usr/bin/env python3
"""Full corpus ingestion pipeline for Egyptian Civil Code.

Thin wrapper around ragapp.offline_ingestion.ingestion for CLI usage.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient

from ragapp.core.config import get_settings
from ragapp.modules.rag.domain.adapters.retrievers import QdrantRetriever
from ragapp.modules.rag.offline_ingestion.ingestion import run_ingestion

DEFAULT_CHUNKS_PATH = Path("data/corpus/processed/chunks.article_aware.jsonl")
DEFAULT_ALIAS = "egyptian-civil-code"
DEFAULT_QDRANT_URL = "http://localhost:6333"
BGE_M3_DIM = 1024


DEFAULT_CHUNKS_PATH = Path("data/corpus/processed/chunks.article_aware.jsonl")
DEFAULT_ALIAS = "egyptian-civil-code"
DEFAULT_QDRANT_URL = "http://localhost:6333"
BGE_M3_DIM = 1024


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Egyptian Civil Code corpus into Qdrant"
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks JSONL file",
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

    settings = get_settings()

    print("=== Egyptian Civil Code Corpus Ingestion ===")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Target alias: {args.alias}")
    print(f"Chunks file: {args.chunks_path}")

    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_path}")

    result = run_ingestion(
        settings=settings,
        qdrant_url=args.qdrant_url,
        alias=args.alias,
        chunks_path=args.chunks_path,
        batch_size=args.batch_size,
        upsert_batch=args.upsert_batch,
    )

    print(f"\nBuild ID: {result.build_id}")
    print(f"Alias: {args.alias} -> {result.build_id}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Build duration: {result.build_duration_ms:.0f}ms")
    print(f"Manifest: {result.manifest_path}")

    if args.verify:
        print("\n=== Running verification query ===")
        client = QdrantClient(url=args.qdrant_url, timeout=30.0)
        retriever = QdrantRetriever(client, args.alias)

        async def test() -> None:
            results = await retriever.retrieve(
                dense=[0.0] * BGE_M3_DIM,
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


if __name__ == "__main__":
    main()
