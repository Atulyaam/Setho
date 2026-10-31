"""Authentication and Token Management Router for Stetho API."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from stetho_api.auth.dependencies import get_current_user, get_user_repository
from stetho_api.auth.tokens import AuthenticationError, create_access_token
from stetho_api.config import Settings, get_settings
from stetho_api.db.user_repo import User, UserRepository
from stetho_api.exceptions import StethoAPIException

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Unique alphanumeric username")
    password: str = Field(..., min_length=8, max_length=128, description="Plaintext password (min 8 chars)")
    role: Optional[str] = Field(default="inspector", description="Self-registration role (strictly restricted to 'inspector')")


class RoleUpdateRequest(BaseModel):
    role: str = Field(..., description="Target role: 'inspector' or 'admin'")


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


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
    description="Creates a new inspector user account with PBKDF2 hashed password. Admin role cannot be self-assigned.",
)
async def register_user(
    request: RegisterRequest,
    user_repo: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    # Privilege escalation defense: Public self-registration cannot grant admin role
    if request.role and request.role.lower() != "inspector":
        raise StethoAPIException(
            "Privilege escalation blocked: Public self-registration cannot grant 'admin' privileges. Role is restricted to 'inspector'.",
            code="PRIVILEGE_ESCALATION_BLOCKED",
            status_code=403,
        )

    user = user_repo.create_user(
        username=request.username,
        password_plain=request.password,
        role="inspector",
    )
    return UserResponse(**user.to_dict())


@router.put(
    "/users/{user_id}/role",
    response_model=UserResponse,
    summary="Update user role (Admin only)",
    description="Enables authorized administrators to manage role permissions.",
)
async def update_user_role(
    user_id: str,
    request: RoleUpdateRequest,
    current_user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repository),
) -> UserResponse:
    if current_user.role != "admin":
        raise StethoAPIException(
            "Access forbidden: Only system administrators can assign or update user roles.",
            code="FORBIDDEN_ADMIN_REQUIRED",
            status_code=403,
        )

    updated_user = user_repo.update_role(user_id, request.role)
    return UserResponse(**updated_user.to_dict())


@router.post(
    "/change-password",
    summary="Change account password",
    description="Rotates account password by verifying current credentials.",
)
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repository),
):
    authenticated = user_repo.authenticate(current_user.username, request.current_password)
    if not authenticated:
        raise StethoAPIException(
            "Incorrect current password.",
            code="INVALID_CREDENTIALS",
            status_code=401,
        )

    user_repo.update_password(current_user.user_id, request.new_password)
    return {"status": "success", "message": "Password updated successfully."}


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

