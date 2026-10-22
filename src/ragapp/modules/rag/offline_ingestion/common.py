"""Shared offline corpus preparation for both retrieval store builders.

The Qdrant store and the BGE index are built from the same chunk JSONL and the
same embedder. Keeping loading, text construction, embedding, and the corpus
revision here guarantees both retrieval arms see identical documents.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from ragapp.core.config import Settings
from ragapp.modules.rag.domain.entities import Chunk
from ragapp.modules.rag.factories import get_embedder


def compute_corpus_revision(chunks_path: Path) -> str:
    """Hash the chunk file that is the actual index input.

    Only `chunks_path` is hashed. Sibling artifacts (`document.json`,
    `pages.jsonl`, and especially an already-built `local_index/`) are not
    inputs and must not change the revision. Returns an empty string when the
    chunks file does not exist.
    """
    if not chunks_path.exists():
        return ""
    return hashlib.md5(chunks_path.read_bytes()).hexdigest()[:12]


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
    """Load JSONL chunk records and convert them to `Chunk` entities."""
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
    """Embed every chunk once with the configured Embedder (dense + sparse).

    Offline-only: closes the shared embedder when finished, so it must not be
    called in the middle of a long-lived request process.
    """
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
