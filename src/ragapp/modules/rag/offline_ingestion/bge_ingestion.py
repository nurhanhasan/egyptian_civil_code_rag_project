"""Build the precomputed local BGE-M3 index (RETRIEVER_BACKEND=bge).

Offline only: embed every corpus chunk once, persist dense + sparse vectors
plus chunk metadata, and record provenance. The request path then loads the
artifact with no model call and no Qdrant.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ragapp.core.config import Settings
from ragapp.modules.rag.domain.adapters.bge_retriever import (
    LocalIndexManifest,
    build_manifest,
    write_local_index,
)
from ragapp.modules.rag.offline_ingestion.common import (
    compute_corpus_revision,
    embed_chunks,
    load_chunks,
)

DEFAULT_CHUNKS_PATH = Path("data/corpus/processed/chunks.article_aware.jsonl")
# Root-level artifact, sibling of qdrant_storage/. It must live OUTSIDE data/
# because .dockerignore excludes data/ wholesale — the image COPY needs it.
DEFAULT_OUT_DIR = Path("bge_storage")


def build_local_index(
    settings: Settings,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    batch_size: int = 32,
) -> LocalIndexManifest:
    """Build the BGE-M3 index from `chunks_path` into `out_dir` (offline).

    Steps:
    1. validate the chunks file and load it into `Chunk` entities;
    2. embed every chunk once with the configured Embedder (dense + sparse);
    3. record embedder/corpus provenance in a manifest;
    4. write the index files (`dense.npy`, `sparse.jsonl`, `chunks.jsonl`,
       `manifest.json`).

    No Qdrant or network service is involved. Returns the manifest.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    embedded = asyncio.run(embed_chunks(chunks, settings, batch_size=batch_size))
    dense = [vector for _, vector, _ in embedded]
    sparse = [weights for _, _, weights in embedded]

    manifest = build_manifest(
        embedder_model_id=settings.BGE_M3_MODEL_PATH,
        embedder_revision_id=settings.EMBEDDER_MODEL_REVISION_ID,
        dim=len(dense[0]) if dense else 0,
        count=len(chunks),
        rrf_k=settings.RETRIEVER_RRF_K,
        corpus_revision=compute_corpus_revision(chunks_path),
        source_chunks_path=str(chunks_path),
    )
    write_local_index(out_dir, chunks, dense, sparse, manifest)
    return manifest
