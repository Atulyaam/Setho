"""Exception definitions and FastAPI exception handlers."""

import logging
from typing import Any, Optional
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from stetho_api.schemas.common import ErrorDetail, ErrorResponse, get_utc_now

logger = logging.getLogger("stetho")


class StethoAPIException(Exception):
    """Base exception class for custom API errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class ModelNotReadyException(StethoAPIException):
    """Raised when an inference request is made but model is not available."""

    def __init__(self, message: str = "Machine model baseline is not ready"):
        super().__init__(
            message=message,
            code="MODEL_NOT_READY",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ResourceNotFoundException(StethoAPIException):
    """Raised when a machine or analysis record is not found."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} '{identifier}' not found",
            code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ValidationException(StethoAPIException):
    """Raised when domain business validation fails."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class ConflictException(StethoAPIException):
    """Raised when attempting to create a duplicate resource."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


def get_request_id(request: Request) -> Optional[str]:
    """Extract correlation request ID from request state if available."""
    return getattr(request.state, "request_id", None)


async def stetho_exception_handler(request: Request, exc: StethoAPIException) -> JSONResponse:
    """Handle application-specific exceptions."""
    req_id = get_request_id(request)
    logger.warning(f"Handled application exception: {exc.code} - {exc.message} (request_id={req_id})")
    error_payload = ErrorResponse(
        error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details),
        request_id=req_id,
        timestamp=get_utc_now(),
    )
    return JSONResponse(status_code=exc.status_code, content=error_payload.model_dump())


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request payload validation errors (422)."""
    req_id = get_request_id(request)
    logger.info(f"Validation error: {exc.errors()} (request_id={req_id})")
    error_payload = ErrorResponse(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request input validation failed",
            details=jsonable_encoder(exc.errors()),
        ),
        request_id=req_id,
        timestamp=get_utc_now(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(error_payload.model_dump()),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions."""
    req_id = get_request_id(request)
    code = f"HTTP_{exc.status_code}"
    error_payload = ErrorResponse(
        error=ErrorDetail(code=code, message=str(exc.detail)),
        request_id=req_id,
        timestamp=get_utc_now(),
    )
    return JSONResponse(status_code=exc.status_code, content=error_payload.model_dump())


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected server exceptions."""
    req_id = get_request_id(request)
    logger.error(f"Unhandled server error: {exc} (request_id={req_id})", exc_info=True)
    error_payload = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred. Please try again later.",
        ),
        request_id=req_id,
        timestamp=get_utc_now(),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload.model_dump(),
    )

