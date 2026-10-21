from fastapi import Depends

from ragapp.core.config import get_settings
from ragapp.modules.rag.factories import get_retriever


class HealthCheck:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    async def health_check(self):
        """
        /health returns {status: healthy, documents_indexed: N}
        """

        documents_indexed = 0
        client = None
        try:
            # TODO: Check for Retriver class and call points_count
            client = get_retriever(self.settings)
            info = client.get_collection(self.settings.QDRANT_COLLECTION_ALIAS)
            documents_indexed = info.points_count
        except Exception:
            documents_indexed = 0

        return {"status": "healthy", "documents_indexed": documents_indexed}


def get_health_check(settings=Depends(get_settings)) -> HealthCheck:
    return HealthCheck(settings=settings)
