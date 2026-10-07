"""Distributed token-bucket rate limiting.

Strategy pattern (same shape as core/cache.py):
  RateLimiter       -> Context: the HTTP layer calls .allow(key)
   _RedisBackend     -> ConcreteStrategy: distributed state on RATELIMIT_DB (db 3)
   _InMemoryBackend  -> ConcreteStrategy: per-process counter (tests / single-node)

Every replica runs the same atomic Lua script against the same counter in Redis,
so the global cap is NOT N x per-replica.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC
from dataclasses import dataclass
from typing import Optional

from ragapp.core.config import Settings

try:
    import redis as _redis
except Exception:
    _redis = None

logger = logging.getLogger(__name__)

# Redis atomic token-bucket Lua script.
#   KEYS[1] = bucket hash key
#   ARGV[1..4] = capacity, period, ttl, now (seconds, float for determinism)
# Returns: {allowed(1/0), retry_after(int seconds), remaining(tokens, int)}
_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local period = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local last = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    last = now
end

local rate = capacity / period
local elapsed = now - last
if elapsed < 0 then elapsed = 0 end
tokens = math.min(capacity, tokens + elapsed * rate)
last = now

local allowed = 0
local retry_after = 0

if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
else
    retry_after = math.ceil((1 - tokens) / rate)
end

redis.call('HSET', key, 'tokens', tokens, 'ts', last)
redis.call('EXPIRE', key, ttl)

return {allowed, retry_after, math.floor(tokens)}
"""


@dataclass
class RateDecision:
    allowed: bool
    retry_after: int
    limit: int = 0
    remaining: int = 0
    backend: str = "inmemory"


class RateLimitBackend(ABC):
    def allow(
        self,
        key: str,
        capacity: int,
        refill_period: int,
        ttl: int,
        now: Optional[float] = None,
    ) -> RateDecision:
        raise NotImplementedError


class _InMemoryBackend(RateLimitBackend):
    """Per-process. Not distributed - N replicas = N x cap"""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._buckets: dict[str, list] = {}

    def allow(
        self,
        key: str,
        capacity: int,
        refill_period: int,
        ttl: int,
        now: Optional[float] = None,
    ) -> RateDecision:
        t = float(now if now is not None else time.time())
        rate = capacity / refill_period
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                tokens, last = float(capacity), t
            else:
                tokens, last = float(bucket[0]), float(bucket[1])
                elapsed = max(0.0, t - last)
                tokens = min(capacity, tokens + elapsed * rate)
                last = t
            allowed = False
            retry_after = 0
            if tokens >= 1:
                tokens -= 1
                allowed = True
            else:
                retry_after = max(1, int(math.ceil((1 - tokens) / rate)))
            self._buckets[key] = [tokens, last]
        return RateDecision(
            allowed=allowed,
            retry_after=retry_after,
            limit=capacity,
            remaining=int(tokens) + (1 if allowed else 0),
            backend="inmemory",
        )


class _RedisBackend(RateLimitBackend):
    """Distributed token bucket on RATELIMIT_DB via an atomic Lua EVAL."""

    def __init__(self, url: str, ttl_default: int = 120) -> None:
        if _redis is None:
            raise RuntimeError("redis package is not installed")
        self._client = _redis.Redis.from_url(url, decode_responses=True)
        self._ttl_default = ttl_default
        self._script = self._client.register_script(_LUA_TOKEN_BUCKET)

    def allow(
        self,
        key: str,
        capacity: int,
        refill_period: int,
        ttl: int,
        now: Optional[float] = None,
    ) -> RateDecision:
        t = float(now if now is not None else self._client.time()[0])
        tt = ttl or self._ttl_default
        bucket_key = f"rl:{key}"
        try:
            result = self._script(
                [bucket_key],
                [str(capacity), str(refill_period), str(tt), str(t)],
            )
            if result is None or len(result) < 3:
                result = self._client.eval(
                    _LUA_TOKEN_BUCKET,
                    1,
                    bucket_key,
                    str(capacity),
                    str(refill_period),
                    str(tt),
                    str(t),
                )
        except Exception:
            # Fail open: don't take the whole API down if Redis blipped.
            logger.warning("rate_limit_degraded: Redis unreachable for key=%s", key)
            return RateDecision(
                allowed=True,
                retry_after=0,
                limit=capacity,
                remaining=-1,
                backend="redis",
            )
        vals = [int(v) for v in (result or [1, 0, 0])]
        allowed, retry_after, remaining = vals[0], vals[1], vals[2]
        return RateDecision(
            allowed=bool(allowed),
            retry_after=max(0, retry_after),
            limit=capacity,
            remaining=remaining,
            backend="redis",
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class RateLimiter:
    def __init__(
        self,
        backend: RateLimitBackend,
        capacity: int,
        refill_period: int,
        ttl: int,
    ) -> None:
        self._backend = backend
        self._capacity = capacity
        self._refill_period = refill_period
        self._ttl = ttl

    def allow(self, key: str, now: Optional[float] = None) -> RateDecision:
        """
        Check if a request is allowed under the rate limit.

        ARG:
            key: str - the identity used to bucket requests (e.g. client IP or API key)
            now: Optional[float] - current time in sec (float) for determinism in tests
        """
        return self._backend.allow(
            key,
            self._capacity,
            self._refill_period,
            self._ttl,
            now,
        )

    @property
    def backend_kind(self) -> str:
        return self._backend.__class__.__name__


# Memoize by settings-instance id so the bucket persists across requests yet
# refreshes when settings change (clear settings cache -> new instance -> new limiter).
_LIMITER_CACHE: dict[int, object] = {}


def get_rate_limiter(settings: Settings) -> Optional[RateLimiter]:
    if not settings.RATE_LIMIT_ENABLED:
        return None

    cached = _LIMITER_CACHE.get(id(settings))
    if cached is not None:
        return cached

    capacity = max(1, settings.RATE_LIMIT_CAPACITY)
    period = max(1, settings.RATE_LIMIT_REFILL_PERIOD)
    ttl = settings.RATE_LIMIT_BUCKET_TTL or max(2 * period, 60)

    if settings.ENVIRONMENT == "test":
        backend: RateLimitBackend = _InMemoryBackend()
    else:
        backend = _RedisBackend(settings.REDIS_RATELIMIT_URL, ttl_default=ttl)

    limiter = RateLimiter(
        backend=backend,
        capacity=capacity,
        refill_period=period,
        ttl=ttl,
    )
    _LIMITER_CACHE[id(settings)] = limiter
    return limiter


def clear_rate_limiter() -> None:
    _LIMITER_CACHE.clear()
