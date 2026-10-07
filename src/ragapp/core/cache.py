"""
Strategy Pattern

CacheManager     → context
_RedisBackend    → ConcreteStrategy A (production)
_MemoryBackend   → ConcreteStrategy B (testing)
"""

import functools
import inspect
import json
import logging
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import redis.asyncio as asyncredis

from ragapp.core.config import Settings, get_settings

T = TypeVar("T")  # T can be any type, but consistent


logger = logging.getLogger(__name__)


@runtime_checkable
class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def ping(self) -> None: ...
    async def close(self) -> None: ...


@runtime_checkable
class SetOperations(Protocol):
    """Optional set operations - separate capability"""

    async def sadd(self, key: str, *members: str) -> None: ...
    async def smembers(self, key: str) -> set: ...


class _RedisBackend(CacheBackend, SetOperations):
    """
    Redis backend for production.
    """

    def __init__(self, redis_url: str, db: Optional[int] = None):
        self.redis = asyncredis.Redis.from_url(
            redis_url,
            decode_responses=True,  # returns str instead of bytes automatically
            db=db,
        )

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if ttl is not None:
            await self.redis.setex(key, ttl, value)
        else:
            await self.redis.set(key, value)

    async def get(self, key: str) -> Optional[Any]:
        return await self.redis.get(key)

    async def delete(self, key: str) -> None:
        await self.redis.delete(key)

    async def exists(self, key: str) -> bool:
        return await self.redis.exists(key) > 0

    async def sadd(self, key: str, *members: str):
        """
        Adds one or more "unique" values to a Redis Set.
        Usage:
        backend.sadd("user:42:sessions", "token1")
        backend.sadd("user:42:sessions", "token2")

        Redis internally stores:
        user:42:sessions = {
            "token1",
            "token2"
        }

        Why Is This Useful?
            - logout all devices
            - revoke all sessions
            - see active devices
            - invalidate all refresh tokens after password reset
        """
        await self.redis.sadd(key, *members)

    async def smembers(self, key: str) -> set:
        """
        Return all members inside the set.

        Usage:
            backend.smembers("user:42:sessions")

        Returns:
            {"token1", "token2"}
        """
        result = await self.redis.smembers(key)
        return result or set()

    async def ping(self) -> None:
        """Check whether Redis is reachable/alive."""
        await self.redis.ping()

    async def close(self) -> None:
        """Cleanly close the Redis connection/pool."""
        await self.redis.aclose()


class _MemoryBackend(CacheBackend, SetOperations):
    """
    In-memory backend for testing.

    For Unit Tests:
      App -> MemoryBackend -> Python dict

    Although _MemoryBackend has no real I/O, using async to
    implement the "implicit" interface.
    """

    def __init__(self):
        self._data = {}
        self._namespaces = {}

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """
        Set the value with an optional TTL. If TTL is 0 or negative,
        treat as expired immediately.
        """
        if ttl is not None and ttl <= 0:
            expiry = time.monotonic() - 1
        else:
            expiry = time.monotonic() + ttl if ttl is not None else None
        self._data[key] = (value, expiry)

    async def get(self, key: str) -> Optional[Any]:
        """
        Get if not expired, else None.
        Lazy Expiration, keys expire only when accessed.
        """
        if key in self._data:
            value, expiry = self._data[key]
            if expiry is not None and expiry <= time.monotonic():
                del self._data[key]
                return None
            return value
        return None

    async def delete(self, key: str) -> None:
        if key in self._data:
            del self._data[key]

    async def exists(self, key: str) -> bool:
        check = await self.get(key)
        if check is not None:
            return True
        return False

    async def sadd(self, key: str, *members: str) -> None:
        """
        Adds one or more "unique" values to a Redis Set.
        Usage:
        backend.sadd("user:42:sessions", "token1")
        backend.sadd("user:42:sessions", "token2")

        Redis internally stores:
        user:42:sessions = {
            "token1",
            "token2"
        }

        Why Is This Useful?
            - logout all devices
            - revoke all sessions
            - see active devices
            - invalidate all refresh tokens after password reset
        """
        if key not in self._namespaces:
            self._namespaces[key] = set()
        self._namespaces[key].update(members)

    async def smembers(self, key: str) -> set:
        """
        Return all members inside the set.

        Usage:
            backend.smembers("user:42:sessions")

        Returns:
            {"token1", "token2"}
        """
        return self._namespaces.get(key, set()) or set()

    async def ping(self) -> None:
        pass

    async def close(self) -> None:
        pass


class CacheManager:
    """
    Context class in the Strategy Pattern.

    Swaps between two backends depending on environment:
    - _RedisAdminBackend  → production (real Redis)
    - _MemoryBackend      → testing (in-memory, no Redis needed)

    CacheManager never changes regardless of which backend is active.
    Both backends implement the same implicit async interface.
    """

    def __init__(
        self,
        backend: CacheBackend,
        namespace: Optional[str] = None,
        default_ttl: Optional[int] = None,
        settings: Optional[Settings] = None,
    ):
        self._backend = backend
        self.namespace = (
            namespace if namespace is not None else settings.CACHE_NAMESPACE
        )
        self.default_ttl = (
            default_ttl if default_ttl is not None else settings.CACHE_TTL
        )
        self.settings = settings if settings is not None else get_settings()

        backend_type = type(self._backend).__name__
        logger.info(f"CacheManager initialized with backend: {backend_type}")

    def _full_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def _encode(self, value: Any) -> str:
        """
        Convert Python data to a string, so that the backend can store it.
        This works only for JSON-serializable values (dict, list, str, int, bool,
        None, etc.).
        Custom objects need custom serialization.
        """
        return json.dumps(value)

    def _decode(self, value: str) -> Any:
        """On read: json.loads(raw) converts the string back to Python data."""
        return json.loads(value)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Encode and set a value by logical key."""
        full_key = self._full_key(key)
        encoded = self._encode(value)
        effective_ttl = ttl if ttl is not None else self.default_ttl
        await self._backend.set(full_key, encoded, effective_ttl)

        identifier = key.split(":", 1)[0]
        # Namespace = leading segment ("blacklist" from "blacklist:token:42"),
        # matching the exact prefix ``invalidate(pattern)`` looks up in __ns__.
        await self._register_key(identifier, full_key)

    async def get(self, key: str) -> Optional[Any]:
        """Get and decode a value by logical key."""
        full_key = self._full_key(key)
        raw = await self._backend.get(full_key)
        if raw is None:
            return None
        return self._decode(raw)

    async def delete(self, key: str) -> None:
        """Delete a value by logical key."""
        await self._backend.delete(self._full_key(key))

    async def exists(self, key: str) -> bool:
        """Check if logical key exists in cache."""
        return await self._backend.exists(self._full_key(key))

    async def sadd(self, key: str, *members: str) -> None:
        if isinstance(self._backend, SetOperations):
            return await self._backend.sadd(key, *members)

    async def smembers(self, key: str | None = None) -> set:
        if isinstance(self._backend, SetOperations):
            result = await self._backend.smembers(key)
            return result

    async def _register_key(self, identifier: str, full_key: str):
        """
        Using the "Secondary Indexing in Cache Layer" system design to track
        keys by namespace for efficient invalidation.

        The index lives at ``{namespace}:__ns__:{identifier}`` so
        ``invalidate(pattern)`` can look it up by the same shape. Storing the
        index *outside* the normal key space (the ``__ns__`` marker) keeps it
        from colliding with application keys.

        Usage:
        identifier = "blacklist"
        full_key   = "cache:auth:blacklist:token:123"

        Result:
        cache:__ns__:blacklist -> { "cache:auth:blacklist:token:123" }
        """
        if isinstance(self._backend, SetOperations):
            index_key = f"{self.namespace}:__ns__:{identifier}"
            await self._backend.sadd(index_key, full_key)

    def _make_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        func_name = func.__qualname__
        args_str = "_".join(str(arg) for arg in args)
        kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{func_name}:{args_str}:{kwargs_str}"

    async def get_or_set(
        self,
        key: str,
        # loader returns either T (sync) or Awaitable[T] (async).
        loader: Callable[[], Union[T, Awaitable[T]]],
        ttl: Optional[int] = None,
    ) -> T:
        """
        Returns cached value if exists, otherwise calls loader(), caches the result
        and returns it.
        Leveraging the Cache-Aside (Lazy Loading) pattern.
        Compose, don't extend.
        """
        full_key = self._full_key(key)
        raw = await self._backend.get(full_key)
        if raw is not None:
            return self._decode(raw)

        value = loader()
        if inspect.isawaitable(value):
            value = await value
        await self.set(key, value, ttl)

        return value

    async def invalidate(self, pattern: str):
        """Deletes all cached keys under a namespace prefix."""
        if isinstance(self._backend, SetOperations):
            ns_key = f"{self.namespace}:__ns__:{pattern}"
            keys = await self._backend.smembers(ns_key)
            for key in keys:
                await self._backend.delete(key)
            await self._backend.delete(ns_key)

    def cached(self, ttl: Optional[int] = None, key_builder: Optional[Callable] = None):
        """
        A decorator that automatically caches any function's return value.

        Usage:
        @cache_manager.cached(ttl=300)
        async def get_recipes():
            ...
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if key_builder:
                    key = key_builder(func, args, kwargs)
                else:
                    key = self._make_key(func, args, kwargs)
                return await self.get_or_set(key, lambda: func(*args, **kwargs), ttl)

            return wrapper

        return decorator

    async def init(self):
        await self._backend.ping()

    async def close(self):
        await self._backend.close()


def get_cache_manager(settings: Settings) -> CacheManager:
    """
    Production/dev talk to Redis; ``test`` uses the in-memory backend so unit
    and orchestration tests need no Redis.
    """
    if settings.ENVIRONMENT == "test":
        backend: CacheBackend = _MemoryBackend()
    else:
        # Point at the cache logical database (db 0) per MLOPS_TODO #8.
        backend = _RedisBackend(settings.REDIS_CACHE_URL)
    return CacheManager(backend=backend, settings=settings)
