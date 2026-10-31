"""User Repository for Authentication and Identity Management."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional
import uuid

from stetho_api.auth.password import hash_password, verify_password
from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import StethoAPIException

logger = logging.getLogger("stetho.user_repo")


@dataclass
class User:
    user_id: str
    username: str
    hashed_password: str
    role: str  # "admin" or "inspector"
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
        }


class UserRepository:
    """Data access repository for user accounts."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_by_id(self, user_id: str) -> Optional[User]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?;", (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return User(
            user_id=row["user_id"],
            username=row["username"],
            hashed_password=row["hashed_password"],
            role=row["role"],
            created_at=row["created_at"],
        )

    def get_by_username(self, username: str) -> Optional[User]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?;", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        return User(
            user_id=row["user_id"],
            username=row["username"],
            hashed_password=row["hashed_password"],
            role=row["role"],
            created_at=row["created_at"],
        )

    def create_user(self, username: str, password_plain: str, role: str = "inspector") -> User:
        if self.get_by_username(username):
            raise StethoAPIException(
                f"Username '{username}' already exists",
                code="USERNAME_EXISTS",
                status_code=409,
            )

        user_id = str(uuid.uuid4())
        hashed_pw = hash_password(password_plain)
        now = get_utc_now_iso()

        with self.db.transaction() as cursor:
            cursor.execute("""
            INSERT INTO users (user_id, username, hashed_password, role, created_at)
            VALUES (?, ?, ?, ?, ?);
            """, (user_id, username, hashed_pw, role, now))

        return User(
            user_id=user_id,
            username=username,
            hashed_password=hashed_pw,
            role=role,
            created_at=now,
        )

    def authenticate(self, username: str, password_plain: str) -> Optional[User]:
        user = self.get_by_username(username)
        if not user:
            return None
        if not verify_password(password_plain, user.hashed_password):
            return None
        return user

