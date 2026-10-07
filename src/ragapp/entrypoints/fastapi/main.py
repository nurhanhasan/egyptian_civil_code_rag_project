import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI

from ragapp.core.config import get_settings
from ragapp.core.logging import setup_logging
from ragapp.dependencies.cache import get_cache
from ragapp.dependencies.rate_limit import require_rate_limit
from ragapp.exceptions.handlers import register_exception_handlers
from ragapp.middleware.cors import setup_cors
from ragapp.middleware.logging import LoggingMiddleware
from ragapp.modules.health.router import router as health_router
from ragapp.modules.rag.router import router as rag_router

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Start logging
    setup_logging()

    # Startup logging
    logger.info("Starting up RAG API...")

    # Initialize cache
    cache_manager = get_cache()
    await cache_manager.init()

    # Start app
    yield

    # Shutdown logging
    logger.info("Shutting down RAG API...")

    # Close cache
    await cache_manager.close()


def create_app() -> FastAPI:

    print(f"Environment: {settings.ENVIRONMENT}")

    # Instantiate FastAPI app
    app = FastAPI(
        title="RAG API",
        version="0.1.0",
        docs_url="/docs",
        lifespan=lifespan,
    )

    # Wire CORS
    setup_cors(app, settings)

    # Wire Logging middleware
    app.add_middleware(LoggingMiddleware)

    # Global router level Atomic Token Bucket Rate Limiting,
    # all routes under /api/v1 are rate-limited.
    # TODO: RATE_LIMIT_ENABLED is disabled by default, enable before commit.
    api_router = APIRouter(
        prefix="/api/v1",
        dependencies=[Depends(require_rate_limit)],
    )

    # Add individual routers
    api_router.include_router(health_router)
    api_router.include_router(rag_router)

    # Register routers
    app.include_router(api_router)

    # Register exception handlers
    register_exception_handlers(app)

    return app


app = create_app()
