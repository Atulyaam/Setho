"""Unit tests for Authentication, Authorization, and Multi-Tenant Isolation (Task 4.5.3)."""

import time
from fastapi.testclient import TestClient
import pytest

from stetho_api.auth.password import hash_password, verify_password
from stetho_api.auth.tokens import create_access_token, decode_access_token
from stetho_api.main import app

client = TestClient(app)


def test_password_hashing_and_verification():
    """Verify PBKDF2-HMAC-SHA256 password hashing and constant-time verification."""
    raw_pw = "SuperSecretIndustrialKey2026!"
    hashed = hash_password(raw_pw)
    assert hashed.startswith("pbkdf2_sha256$100000$")
    assert verify_password(raw_pw, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False
    assert verify_password("", hashed) is False


def test_token_creation_and_expiration():
    """Verify cryptographically signed token creation, decoding, and expiration check."""
    secret = "test-secret-key-for-token-generation-32-chars!"
    payload = {"sub": "usr-123", "username": "alice", "role": "inspector"}

    # Valid token
    tok = create_access_token(payload, secret_key=secret, expires_in_seconds=60)
    claims = decode_access_token(tok, secret_key=secret)
    assert claims["sub"] == "usr-123"
    assert claims["username"] == "alice"
    assert claims["role"] == "inspector"

    # Expired token
    exp_tok = create_access_token(payload, secret_key=secret, expires_in_seconds=-10)
    with pytest.raises(Exception) as exc:
        decode_access_token(exp_tok, secret_key=secret)
    assert "expired" in str(exc.value).lower()


def test_user_registration_and_login_flow():
    """Verify registering a new user, logging in, and retrieving /auth/me."""
    import uuid
    uname = f"tech_{uuid.uuid4().hex[:6]}"
    pw = "SecureInspectorPass2026!"

    # 1. Register
    reg_res = client.post("/api/v1/auth/register", json={
        "username": uname,
        "password": pw,
        "role": "inspector",
    })
    assert reg_res.status_code == 201
    user_info = reg_res.json()
    assert user_info["username"] == uname
    assert user_info["role"] == "inspector"

    # 2. Login
    login_res = client.post("/api/v1/auth/login", json={
        "username": uname,
        "password": pw,
    })
    assert login_res.status_code == 200
    tok_data = login_res.json()
    assert "access_token" in tok_data
    token = tok_data["access_token"]

    # 3. Retrieve /auth/me with Bearer token
    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == uname


def test_unauthenticated_request_rejected():
    """Verify accessing protected routes without Bearer token returns 401."""
    res = client.get("/api/v1/machines")
    assert res.status_code == 401
    assert "Missing Bearer" in res.json()["error"]["message"]


def test_invalid_token_rejected():
    """Verify malformed or forged Bearer token returns 401."""
    res = client.get("/api/v1/machines", headers={"Authorization": "Bearer forged.invalid.token"})
    assert res.status_code == 401


def test_cross_user_resource_isolation():
    """Verify User B cannot access or modify User A's machines (403 Forbidden)."""
    import uuid
    # Create User A
    u_a = f"alice_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/auth/register", json={"username": u_a, "password": "Password123!"})
    tok_a = client.post("/api/v1/auth/login", json={"username": u_a, "password": "Password123!"}).json()["access_token"]

    # Create User B
    u_b = f"bob_{uuid.uuid4().hex[:6]}"
    client.post("/api/v1/auth/register", json={"username": u_b, "password": "Password123!"})
    tok_b = client.post("/api/v1/auth/login", json={"username": u_b, "password": "Password123!"}).json()["access_token"]

    # User A creates machine
    m_id = f"machine_alice_{uuid.uuid4().hex[:6]}"
    create_res = client.post(
        "/api/v1/machines",
        json={"machine_id": m_id, "name": "Alice Private Machine", "category": "fan"},
        headers={"Authorization": f"Bearer {tok_a}"},
    )
    assert create_res.status_code == 201

    # User A accesses machine -> 200 OK
    res_a = client.get(f"/api/v1/machines/{m_id}", headers={"Authorization": f"Bearer {tok_a}"})
    assert res_a.status_code == 200

    # User B attempts to access User A's machine -> 403 Forbidden!
    res_b = client.get(f"/api/v1/machines/{m_id}", headers={"Authorization": f"Bearer {tok_b}"})
    assert res_b.status_code == 403
    assert "Access denied" in res_b.json()["error"]["message"]

    # Admin accesses User A's machine -> 200 OK (Admin override)
    tok_admin = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StethoAdmin2026!"},
    ).json()["access_token"]
    res_admin = client.get(f"/api/v1/machines/{m_id}", headers={"Authorization": f"Bearer {tok_admin}"})
    assert res_admin.status_code == 200

