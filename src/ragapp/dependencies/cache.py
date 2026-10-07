from ragapp.core.cache import CacheManager, get_cache_manager
from ragapp.core.config import get_settings

settings = get_settings()


def get_cache() -> CacheManager:
    return get_cache_manager(settings=settings)
