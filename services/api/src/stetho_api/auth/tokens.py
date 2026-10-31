"""Cryptographically signed, expiring token generator and validator."""

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

from stetho_api.exceptions import StethoAPIException


class AuthenticationError(StethoAPIException):
    """Raised when authentication credentials or token are invalid."""

    def __init__(self, message: str = "Invalid or expired authentication credentials", code: str = "AUTH_FAILED"):
        super().__init__(message=message, code=code, status_code=401)


class AuthorizationError(StethoAPIException):
    """Raised when an authenticated user attempts to access an unauthorized resource."""

    def __init__(self, message: str = "Permission denied for this resource", code: str = "FORBIDDEN"):
        super().__init__(message=message, code=code, status_code=403)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s.encode("utf-8"))


def create_access_token(
    payload: Dict[str, Any],
    secret_key: str,
    expires_in_seconds: int = 3600,
) -> str:
    """Generate a signed, URL-safe access token with an expiration timestamp."""
    exp_ts = int(time.time()) + expires_in_seconds
    token_data = {
        **payload,
        "exp": exp_ts,
        "iat": int(time.time()),
    }
    payload_json = json.dumps(token_data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)

    signature = hmac.new(
        secret_key.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature_b64 = _b64url_encode(signature)

    return f"{payload_b64}.{signature_b64}"


def decode_access_token(token: str, secret_key: str) -> Dict[str, Any]:
    """Verify HMAC signature and expiration timestamp, returning token claims."""
    parts = token.split(".")
    if len(parts) != 2:
        raise AuthenticationError("Malformed authentication token format")

    payload_b64, signature_b64 = parts[0], parts[1]

    # Verify signature
    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    try:
        provided_sig = _b64url_decode(signature_b64)
    except Exception:
        raise AuthenticationError("Invalid token signature encoding")

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise AuthenticationError("Invalid authentication signature")

    # Decode and check expiration
    try:
        payload_bytes = _b64url_decode(payload_b64)
        claims = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        raise AuthenticationError("Corrupted token payload")

    exp_ts = claims.get("exp")
    if not exp_ts or time.time() > exp_ts:
        raise AuthenticationError("Authentication token has expired")

    return claims

