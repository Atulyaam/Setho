"""FastAPI dependencies for Authentication and Authorization."""

from typing import Optional
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from stetho_api.auth.tokens import AuthenticationError, AuthorizationError, decode_access_token
from stetho_api.config import Settings, get_settings
from stetho_api.db.database import get_db
from stetho_api.db.user_repo import User, UserRepository

security = HTTPBearer(auto_error=False)


def get_user_repository(settings: Settings = Depends(get_settings)) -> UserRepository:
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    return UserRepository(db)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    user_repo: UserRepository = Depends(get_user_repository),
    settings: Settings = Depends(get_settings),
) -> User:
    """Validate Bearer access token and return authenticated User domain model."""
    if not credentials or not credentials.credentials:
        raise AuthenticationError("Missing Bearer authentication token")

    token = credentials.credentials
    claims = decode_access_token(token, secret_key=settings.SECRET_KEY)

    user_id = claims.get("sub")
    if not user_id:
        raise AuthenticationError("Token payload missing subject identifier")

    user = user_repo.get_by_id(user_id)
    if not user:
        raise AuthenticationError("Authenticated user record no longer exists")

    return user


async def get_current_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Verify that current user possesses administrator privileges."""
    if current_user.role != "admin":
        raise AuthorizationError("Administrator privilege required for this operation")
    return current_user

