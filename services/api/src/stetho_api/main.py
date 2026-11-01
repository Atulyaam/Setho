"""FastAPI application entry point for Stetho."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from stetho_api.config import get_settings
from stetho_api.exceptions import (
    StethoAPIException,
    generic_exception_handler,
    http_exception_handler,
    stetho_exception_handler,
    validation_exception_handler,
)
from stetho_api.logging import RequestLoggingMiddleware, logger, setup_logging
from stetho_api.routers.api_v1 import router as api_v1_router
from stetho_api.routers.assets import router as assets_router
from stetho_api.routers.health import router as health_router
from stetho_api.routers.inference import router as inference_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook for startup and shutdown procedures."""
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
    logger.info(
        f"Starting {settings.PROJECT_NAME} v{settings.VERSION} "
        f"[env={settings.ENVIRONMENT}, debug={settings.DEBUG}]"
    )

    # Initialize storage directories and database schema
    settings.STORAGE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    from stetho_api.db.database import get_db
    get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")

    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Machine-specific acoustic anomaly monitoring and inspection API.",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    # Middleware: Request Correlation & Structured Logging
    application.add_middleware(RequestLoggingMiddleware)

    # Middleware: CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    application.add_exception_handler(StethoAPIException, stetho_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(Exception, generic_exception_handler)

    # Register Routers
    application.include_router(health_router)
    application.include_router(api_v1_router, prefix=settings.API_PREFIX)
    # Direct /api route aliases for Sprint 7 endpoints
    application.include_router(assets_router, prefix=settings.API_PREFIX)
    application.include_router(inference_router, prefix=settings.API_PREFIX)

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "stetho_api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
    )
