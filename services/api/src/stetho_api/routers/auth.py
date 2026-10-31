"""Authentication and Token Management Router for Stetho API."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from stetho_api.auth.dependencies import get_current_user, get_user_repository
from stetho_api.auth.tokens import AuthenticationError, create_access_token
from stetho_api.config import Settings, get_settings
from stetho_api.db.user_repo import User, UserRepository

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Unique alphanumeric username")
    password: str = Field(..., min_length=8, max_length=128, description="Plaintext password (min 8 chars)")
    role: Optional[str] = Field(default="inspector", description="Role: 'inspector' or 'admin'")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: Dict[str, Any]


class UserResponse(BaseModel):
    user_id: str
    username: str
    role: str
    created_at: str


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Creates a new inspector or administrator user account with PBKDF2 hashed password.",
)
async def register_user(
    request: RegisterRequest,
    user_repo: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    user = user_repo.create_user(
        username=request.username,
        password_plain=request.password,
        role=request.role or "inspector",
    )
    return UserResponse(**user.to_dict())


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticates credentials and returns a signed, expiring cryptographic bearer token.",
)
async def login_user(
    request: LoginRequest,
    user_repo: UserRepository = Depends(get_user_repository),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    user = user_repo.authenticate(request.username, request.password)
    if not user:
        raise AuthenticationError("Incorrect username or password")

    expires_in = 86400  # 24 hours
    token = create_access_token(
        payload={"sub": user.user_id, "username": user.username, "role": user.role},
        secret_key=settings.SECRET_KEY,
        expires_in_seconds=expires_in,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_seconds=expires_in,
        user=user.to_dict(),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Current user profile",
    description="Returns the profile and role of the currently authenticated bearer token holder.",
)
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(**current_user.to_dict())

