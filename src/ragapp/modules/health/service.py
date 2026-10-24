import json
import logging
from pathlib import Path

from fastapi import Depends

from ragapp.core.config import get_settings
from ragapp.modules.rag.factories import get_qdrant_client

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("qdrant", "bge")


class HealthCheck:
    """Readiness probe that knows *which* retrieval arm this process serves.

    `retriever_backend` is the point: one image serves qdrant and bge.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def _documents_indexed(self, backend: str) -> int:
        """Number of chunks the configured store actually holds."""
        if backend == "bge":
            index_dir = Path(self.settings.BGE_RETRIEVER_PATH)
            manifest = json.loads(
                (index_dir / "manifest.json").read_text(encoding="utf-8")
            )
            return int(manifest["count"])

        # (QdrantLocal is single-writer per process).
        # Count through the shared client the factory owns
        client = get_qdrant_client(self.settings)
        info = client.get_collection(self.settings.QDRANT_COLLECTION_ALIAS)
        return int(info.points_count or 0)

    async def health_check(self):
        """
        /health returns {status, retriever_backend, documents_indexed}.

        A broken or empty store reports `status="degraded"` while still
        returning HTTP 200 — the compose healthcheck gates on the status code,
        so the body has to carry the truth.
        """
        backend = (self.settings.RETRIEVER_BACKEND or "").strip().lower()

        if backend not in SUPPORTED_BACKENDS:
            logger.warning("health: unsupported RETRIEVER_BACKEND=%r", backend)
            return {
                "status": "degraded",
                "documents_indexed": 0,
            }

        try:
            documents_indexed = self._documents_indexed(backend)
        except Exception as exc:  # store missing, locked, or unreadable
            logger.warning("health: %s store unreadable: %s", backend, exc)
            return {
                "status": "degraded",
                "documents_indexed": 0,
            }

        return {
            "status": "healthy" if documents_indexed > 0 else "degraded",
            "documents_indexed": documents_indexed,
        }


def get_health_check(settings=Depends(get_settings)) -> HealthCheck:
    return HealthCheck(settings=settings)
