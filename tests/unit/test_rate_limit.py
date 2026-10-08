"""distributed rate limiting.

Always runs (no external deps) for the token-bucket algorithm and the 429+RetryAfter
HTTP path (in-memory backend). The live-Redis "distributed proof" runs when Redis is
reachable on REDIS_RATELIMIT_URL (db 3) and is skipped otherwise, so CI without
Redis still passes.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import redis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragapp.core.rate_limit import (  # noqa: E402
    RateLimiter,
    _InMemoryBackend,
    _RedisBackend,
    get_rate_limiter,
)

RESULTS: list = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def test_token_bucket_math():
    # capacity=3 over 10s window.
    b = _InMemoryBackend()
    lim = RateLimiter(backend=b, capacity=3, refill_period=10, ttl=60)
    d1 = lim.allow("k", now=0.0)
    d2 = lim.allow("k", now=0.0)
    d3 = lim.allow("k", now=0.0)
    check("first N<=capacity allowed", d1.allowed and d2.allowed and d3.allowed)
    d4 = lim.allow("k", now=0.0)
    check("N+1th rejected", (not d4.allowed) and d4.retry_after >= 1)
    check("distinct keys independent", lim.allow("other", now=0.0).allowed)
    # After a full refill window the bucket is full again.
    df = lim.allow("k", now=11.0)
    check("refills after window", df.allowed)


def test_http_429_and_retry_after():
    # With RATE_LIMIT_ENABLED + in-memory backend, a breach raises RateLimitError;
    # the handler maps it to 429 and sets Retry-After.
    os.environ["ENVIRONMENT"] = "test"
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    os.environ["RATE_LIMIT_CAPACITY"] = "3"
    os.environ["RATE_LIMIT_REFILL_PERIOD"] = "10"
    from fastapi.testclient import TestClient

    from ragapp.core.config import get_settings

    settings = get_settings()
    check(
        "factory returns a limiter when enabled",
        get_rate_limiter(settings) is not None,
    )

    # Build a tiny app with the dependency, like main.py wires /api/v1.
    from fastapi import APIRouter, Depends, FastAPI

    from ragapp.dependencies.rate_limit import require_rate_limit

    app = FastAPI()
    app.add_exception_handler(
        __import__(
            "ragapp.exceptions.domain", fromlist=["RateLimitError"]
        ).RateLimitError,
        __import__(
            "ragapp.exceptions.handlers", fromlist=["rate_limit_error_handler"]
        ).rate_limit_error_handler,
    )
    router = APIRouter(dependencies=[Depends(require_rate_limit)])

    @router.get("/ping")
    def ping():
        return {"ok": True}

    app.include_router(router)
    client = TestClient(app)
    cap = settings.RATE_LIMIT_CAPACITY
    statuses = [client.get("/ping").status_code for _ in range(cap)]
    client.get("/ping")  # the (cap+1)th is blocked
    check("first cap requests are 200", all(s == 200 for s in statuses))
    resp = client.get("/ping")  # the (cap+1)th is blocked
    check("breach returns 429", resp.status_code == 429)
    retry = resp.headers.get("Retry-After")
    check("429 carries Retry-After header", retry is not None and retry.isdigit())
    check(
        "429 body is problem+json (RateLimitError)",
        resp.json().get("title") == "RateLimitError",
    )


def test_distributed_proof_live():
    # The whole point of #9: N concurrent workers all hitting ONE Redis counter
    # see a GLOBAL cap, not N x cap.
    from ragapp.core.config import get_settings

    url = get_settings().REDIS_RATELIMIT_URL
    try:
        redis.Redis.from_url(url, decode_responses=True).ping()
    except Exception as e:
        print(f"[SKIP] distributed proof - Redis not reachable at {url} ({e})")
        return
    # Clean the bucket between runs.
    redis.Redis.from_url(url, decode_responses=True).delete("rl:distproof")

    capacity, period = 10, 100
    limiter = RateLimiter(
        backend=_RedisBackend(url, ttl_default=period),
        capacity=capacity,
        refill_period=period,
        ttl=period,
    )
    n_workers = 20
    counter = {"ok": 0, "blocked": 0}
    lock = threading.Lock()

    def one():
        d = limiter.allow("distproof", now=float(limiter._backend._client.time()[0]))
        with lock:
            if d.allowed:
                counter["ok"] += 1
            else:
                counter["blocked"] += 1

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(lambda _: one(), range(n_workers)))

    check(
        "distributed cap: exactly "
        f"{capacity} of {n_workers} concurrent requests allowed",
        counter["ok"] == capacity,
    )
    check(
        f"globally blocked = {n_workers - capacity}",
        counter["blocked"] == n_workers - capacity,
    )
    redis.Redis.from_url(url, decode_responses=True).delete("rl:distproof")


def test_flag_disabled_is_noop():
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    from ragapp.core.config import get_settings

    # get_settings is lru_cached; clear so the new flag takes effect.
    get_settings.cache_clear()
    check(
        "factory returns None when disabled",
        get_rate_limiter(get_settings()) is None,
    )
    get_settings.cache_clear()


if __name__ == "__main__":
    test_token_bucket_math()
    test_http_429_and_retry_after()
    test_distributed_proof_live()
    test_flag_disabled_is_noop()
    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(ok for _, ok in RESULTS) else 1)
