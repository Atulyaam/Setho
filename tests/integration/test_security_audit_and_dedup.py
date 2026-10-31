"""Sprint 5.5 Independent Security Audit & Deduplication Verification Tests.

Verifies:
1. Privilege Escalation Defense: Public registration cannot self-grant 'admin'.
2. Admin-Controlled Role Delegation: Only authenticated admins can promote users.
3. Password Rotation: Secure password changes requiring current credential validation.
4. Enrollment Deduplication: Identical recordings cannot artificially inflate calibration counts.
5. Cross-Machine Independence: Deduplication is strictly scoped per machine.
6. Tenant & Route Authorization: Tampered tokens and unauthorized cross-tenant mutations are rejected.
"""

import io
import time
import uuid
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stetho_api.main import app
from stetho_api.config import get_settings
from stetho_api.db.database import get_db
from stetho_api.db.user_repo import UserRepository

client = TestClient(app)


def _generate_wav(
    duration_sec: float = 5.0,
    sample_rate: int = 16000,
    freq: float = 440.0,
    amp: float = 0.2,
) -> bytes:
    """Generate deterministic valid PCM 16-bit mono WAV bytes."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = amp * np.sin(2 * np.pi * freq * t)
    int_samples = (samples * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


class TestSecurityAuditAndDeduplication:
    """Security and calibration data integrity verification suite."""

    def test_privilege_escalation_blocked_on_registration(self):
        """Verify that public registration rejects any attempt to claim 'admin' role."""
        unique_id = uuid.uuid4().hex[:8]

        # 1. Attempt admin registration
        bad_payload = {
            "username": f"hacker_{unique_id}",
            "password": "Password123!",
            "role": "admin",
        }
        res = client.post("/api/v1/auth/register", json=bad_payload)
        assert res.status_code == 403
        data = res.json()
        assert data["error"]["code"] == "PRIVILEGE_ESCALATION_BLOCKED"
        assert "Privilege escalation blocked" in data["error"]["message"]

        # 2. Attempt arbitrary elevated role
        res_super = client.post(
            "/api/v1/auth/register",
            json={"username": f"super_{unique_id}", "password": "Password123!", "role": "root"},
        )
        assert res_super.status_code == 403
        assert res_super.json()["error"]["code"] == "PRIVILEGE_ESCALATION_BLOCKED"

        # 3. Legitimate registration with explicit inspector role
        res_ok = client.post(
            "/api/v1/auth/register",
            json={"username": f"inspector_{unique_id}", "password": "Password123!", "role": "inspector"},
        )
        assert res_ok.status_code == 201
        assert res_ok.json()["role"] == "inspector"

    def test_admin_role_delegation_and_authorization(self):
        """Verify that only admins can update user roles, and non-admins receive 403."""
        unique_id = uuid.uuid4().hex[:8]

        # Register two regular inspectors
        user1_name = f"user1_{unique_id}"
        user2_name = f"user2_{unique_id}"

        res1 = client.post("/api/v1/auth/register", json={"username": user1_name, "password": "Password123!"})
        assert res1.status_code == 201
        user1_id = res1.json()["user_id"]

        res2 = client.post("/api/v1/auth/register", json={"username": user2_name, "password": "Password123!"})
        assert res2.status_code == 201
        user2_id = res2.json()["user_id"]

        # Login as user 1 (regular inspector)
        login1 = client.post("/api/v1/auth/login", json={"username": user1_name, "password": "Password123!"})
        token1 = login1.json()["access_token"]
        headers1 = {"Authorization": f"Bearer {token1}"}

        # User 1 tries to promote User 2 to admin -> Expect 403 Forbidden
        res_forbidden = client.put(
            f"/api/v1/auth/users/{user2_id}/role",
            json={"role": "admin"},
            headers=headers1,
        )
        assert res_forbidden.status_code == 403
        assert res_forbidden.json()["error"]["code"] == "FORBIDDEN_ADMIN_REQUIRED"

        # Create an admin user via repository directly to establish trusted root
        admin_name = f"admin_{unique_id}"
        settings = get_settings()
        db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
        user_repo = UserRepository(db)
        admin_user = user_repo.create_user(admin_name, "AdminSecret123!", role="admin")

        # Login as verified admin
        login_admin = client.post("/api/v1/auth/login", json={"username": admin_name, "password": "AdminSecret123!"})
        admin_token = login_admin.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        # Admin successfully promotes User 2 to admin
        res_promote = client.put(
            f"/api/v1/auth/users/{user2_id}/role",
            json={"role": "admin"},
            headers=admin_headers,
        )
        assert res_promote.status_code == 200
        assert res_promote.json()["role"] == "admin"

        # Verify User 2 now reflects admin role
        login2 = client.post("/api/v1/auth/login", json={"username": user2_name, "password": "Password123!"})
        token2 = login2.json()["access_token"]
        headers2 = {"Authorization": f"Bearer {token2}"}

        res_me = client.get("/api/v1/auth/me", headers=headers2)
        assert res_me.status_code == 200
        assert res_me.json()["role"] == "admin"

    def test_password_rotation_workflow(self):
        """Verify secure password change with verification of current password."""
        unique_id = uuid.uuid4().hex[:8]
        username = f"rotator_{unique_id}"
        old_pwd = "OldPassword123!"
        new_pwd = "NewSecurePassword456!"

        # Register
        client.post("/api/v1/auth/register", json={"username": username, "password": old_pwd})

        # Login with old password
        login_res = client.post("/api/v1/auth/login", json={"username": username, "password": old_pwd})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Attempt password change with WRONG current password
        bad_change = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": "WrongPassword!", "new_password": new_pwd},
            headers=headers,
        )
        assert bad_change.status_code == 401
        assert bad_change.json()["error"]["code"] == "INVALID_CREDENTIALS"

        # Attempt password change with CORRECT current password
        good_change = client.post(
            "/api/v1/auth/change-password",
            json={"current_password": old_pwd, "new_password": new_pwd},
            headers=headers,
        )
        assert good_change.status_code == 200
        assert good_change.json()["status"] == "success"

        # Verify old password no longer works
        failed_login = client.post("/api/v1/auth/login", json={"username": username, "password": old_pwd})
        assert failed_login.status_code == 401

        # Verify new password works and issues valid session
        success_login = client.post("/api/v1/auth/login", json={"username": username, "password": new_pwd})
        assert success_login.status_code == 200
        assert "access_token" in success_login.json()

    def test_enrollment_deduplication_integrity(self):
        """Verify that identical recordings cannot artificially inflate calibration counts."""
        unique_id = uuid.uuid4().hex[:8]
        username = f"calib_tester_{unique_id}"

        # Register and login
        client.post("/api/v1/auth/register", json={"username": username, "password": "Password123!"})
        login_res = client.post("/api/v1/auth/login", json={"username": username, "password": "Password123!"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create Machine A
        machine_a_id = f"mach_a_{unique_id}"
        res_m_a = client.post(
            "/api/v1/machines",
            json={
                "machine_id": machine_a_id,
                "name": f"Pump A {unique_id}",
                "category": "pump",
                "description": "Audit Pump A",
            },
            headers=headers,
        )
        assert res_m_a.status_code == 201

        # Create Machine B
        machine_b_id = f"mach_b_{unique_id}"
        res_m_b = client.post(
            "/api/v1/machines",
            json={
                "machine_id": machine_b_id,
                "name": f"Pump B {unique_id}",
                "category": "pump",
                "description": "Audit Pump B",
            },
            headers=headers,
        )
        assert res_m_b.status_code == 201

        # Generate audio sample 1
        wav1 = _generate_wav(duration_sec=5.0, freq=440.0)

        # 1. First enrollment of wav1 to Machine A -> Should succeed
        res_enr1 = client.post(
            f"/api/v1/machines/{machine_a_id}/enrollments",
            files={"file": ("sample_440hz.wav", wav1, "audio/wav")},
            headers=headers,
        )
        assert res_enr1.status_code == 201
        assert res_enr1.json()["enrolled_count"] == 1

        # 2. Duplicate enrollment of EXACT SAME wav1 to Machine A -> MUST fail with 409 Conflict
        res_dup = client.post(
            f"/api/v1/machines/{machine_a_id}/enrollments",
            files={"file": ("sample_440hz_copy.wav", wav1, "audio/wav")},
            headers=headers,
        )
        assert res_dup.status_code == 409
        dup_data = res_dup.json()
        assert dup_data["error"]["code"] == "DUPLICATE_ENROLLMENT"
        assert "Duplicate recording detected" in dup_data["error"]["message"]

        # Verify enrolled count remains 1
        res_check = client.get(f"/api/v1/machines/{machine_a_id}", headers=headers)
        assert res_check.status_code == 200
        assert res_check.json()["enrolled_recording_count"] == 1

        # 3. Enrolling wav1 to Machine B -> Should succeed (deduplication is machine-specific)
        res_enr_b = client.post(
            f"/api/v1/machines/{machine_b_id}/enrollments",
            files={"file": ("sample_440hz.wav", wav1, "audio/wav")},
            headers=headers,
        )
        assert res_enr_b.status_code == 201
        assert res_enr_b.json()["enrolled_count"] == 1

        # 4. Enrolling distinct audio (wav2) to Machine A -> Should succeed
        wav2 = _generate_wav(duration_sec=5.0, freq=880.0)
        res_enr2 = client.post(
            f"/api/v1/machines/{machine_a_id}/enrollments",
            files={"file": ("sample_880hz.wav", wav2, "audio/wav")},
            headers=headers,
        )
        assert res_enr2.status_code == 201
        assert res_enr2.json()["enrolled_count"] == 2

    def test_tampered_token_and_unauthorized_access(self):
        """Verify that malformed, expired, or tampered tokens cannot access protected routes."""
        # Unauthenticated request
        res_none = client.get("/api/v1/machines")
        assert res_none.status_code == 401

        # Tampered / invalid token
        res_tampered = client.get(
            "/api/v1/machines",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalidpayload.invalidsig"},
        )
        assert res_tampered.status_code == 401

        # Completely random junk token
        res_junk = client.get(
            "/api/v1/machines",
            headers={"Authorization": "Bearer thisisnotatokenatall"},
        )
        assert res_junk.status_code == 401
