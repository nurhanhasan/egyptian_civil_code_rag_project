#!/usr/bin/env python3
"""CLI for building the local BGE-M3 retrieval index (RETRIEVER_BACKEND=bge).

Thin wrapper around `bge_ingestion.build_local_index`; all indexing logic lives
in the module so it can be reused by CI, tests, and evaluation:

    uv run python -m ragapp.scripts.bge_ingest_corpus [--out-dir bge_storage]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ragapp.core.config import get_settings
from ragapp.modules.rag.offline_ingestion.bge_ingestion import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_OUT_DIR,
    build_local_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the precomputed local BGE-M3 retrieval index"
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the article-aware chunks JSONL",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write the index into",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
    )
    args = parser.parse_args()

    settings = get_settings()

    print("=== Building local BGE-M3 retrieval index ===")
    print(f"Chunks: {args.chunks_path}")
    print(f"Output: {args.out_dir}")
    print(f"Embedder: {settings.BGE_M3_MODEL_PATH}")

    manifest = build_local_index(
        settings=settings,
        chunks_path=args.chunks_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )

    print(f"\nIndexed chunks: {manifest.count}")
    print(f"Dense dim: {manifest.dim}")
    print(f"RRF k: {manifest.rrf_k}")
    print(f"Corpus revision: {manifest.corpus_revision}")
    print(f"Manifest: {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
