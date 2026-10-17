from fastapi import Depends
from qdrant_client import QdrantClient

from ragapp.core.config import get_settings


class HealthCheck:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    async def health_check(self):
        """
        /health returns {status: healthy, documents_indexed: N}
        """

        documents_indexed = 0
        try:
            client = QdrantClient(
                url=self.settings.QDRANT_URL, timeout=self.settings.QDRANT_TIMEOUT
            )
            info = client.get_collection(self.settings.QDRANT_COLLECTION_ALIAS)
            documents_indexed = info.points_count
        except Exception:
            documents_indexed = 0
        finally:
            if client is not None:
                client.close()

        return {"status": "healthy", "documents_indexed": documents_indexed}


def get_health_check(settings=Depends(get_settings)) -> HealthCheck:
    return HealthCheck(settings=settings)
