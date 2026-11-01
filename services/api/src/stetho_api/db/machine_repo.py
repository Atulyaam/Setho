"""Machine Profile Repository for Stetho API."""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional

from stetho_api.auth.tokens import AuthorizationError
from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import ResourceNotFoundException, StethoAPIException

logger = logging.getLogger("stetho.machine_repo")


@dataclass
class MachineProfile:
    """Domain model representing an enrolled physical machine."""
    machine_id: str
    name: str
    category: str
    description: str
    operating_condition: str
    selected_model_policy: str  # "stetho_msae" or "dense_autoencoder"
    enrollment_status: str       # "uncalibrated", "provisional", "calibrated"
    enrolled_recording_count: int
    calibration_status: str
    calibration_profile: Optional[Dict[str, Any]]
    calibration_version: str
    owner_id: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "operating_condition": self.operating_condition,
            "selected_model_policy": self.selected_model_policy,
            "enrollment_status": self.enrollment_status,
            "enrolled_recording_count": self.enrolled_recording_count,
            "calibration_status": self.calibration_status,
            "calibration_version": self.calibration_version,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MachineRepository:
    """Data access repository for machine profiles."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_by_id(self, machine_id: str) -> Optional[MachineProfile]:
        """Fetch a machine profile by its unique identifier."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM machines WHERE machine_id = ?;",
            (machine_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        cal_prof = None
        if "calibration_profile" in row.keys() and row["calibration_profile"]:
            try:
                cal_prof = json.loads(row["calibration_profile"])
            except Exception:
                cal_prof = None

        return MachineProfile(
            machine_id=row["machine_id"],
            name=row["name"],
            category=row["category"],
            description=row["description"] or "",
            operating_condition=row["operating_condition"] or "",
            selected_model_policy=row["selected_model_policy"],
            enrollment_status=row["enrollment_status"],
            enrolled_recording_count=row["enrolled_recording_count"],
            calibration_status=row["calibration_status"],
            calibration_profile=cal_prof,
            calibration_version=row["calibration_version"] if "calibration_version" in row.keys() else "1.0.0",
            owner_id=row["owner_id"] if "owner_id" in row.keys() else "usr_admin",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_or_raise(self, machine_id: str) -> MachineProfile:
        """Fetch machine profile or raise ResourceNotFoundException."""
        machine = self.get_by_id(machine_id)
        if not machine:
            raise ResourceNotFoundException("Machine", machine_id)
        return machine

    def get_authorized(
        self,
        machine_id: str,
        user_id: str,
        is_admin: bool = False,
    ) -> MachineProfile:
        """Fetch machine profile ensuring the requesting user is the owner or an admin."""
        machine = self.get_or_raise(machine_id)
        # Admins can access all machines; benchmark machines owned by usr_admin are accessible to all authenticated users
        if not is_admin and machine.owner_id != user_id and machine.owner_id != "usr_admin":
            raise AuthorizationError(f"Access denied: machine '{machine_id}' is owned by another user.")
        return machine

    def list_all(self, user_id: Optional[str] = None, is_admin: bool = False) -> List[MachineProfile]:
        """List machines. If not admin, filters to user-owned machines plus global benchmarks."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        if is_admin or user_id is None:
            cursor.execute("SELECT * FROM machines ORDER BY machine_id ASC;")
        else:
            cursor.execute(
                "SELECT * FROM machines WHERE owner_id = ? OR owner_id = 'usr_admin' ORDER BY machine_id ASC;",
                (user_id,),
            )
        rows = cursor.fetchall()
        profiles = []
        for r in rows:
            cal_prof = None
            if "calibration_profile" in r.keys() and r["calibration_profile"]:
                try:
                    cal_prof = json.loads(r["calibration_profile"])
                except Exception:
                    cal_prof = None
            profiles.append(
                MachineProfile(
                    machine_id=r["machine_id"],
                    name=r["name"],
                    category=r["category"],
                    description=r["description"] or "",
                    operating_condition=r["operating_condition"] or "",
                    selected_model_policy=r["selected_model_policy"],
                    enrollment_status=r["enrollment_status"],
                    enrolled_recording_count=r["enrolled_recording_count"],
                    calibration_status=r["calibration_status"],
                    calibration_profile=cal_prof,
                    calibration_version=r["calibration_version"] if "calibration_version" in r.keys() else "1.0.0",
                    owner_id=r["owner_id"] if "owner_id" in r.keys() else "usr_admin",
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            )
        return profiles

    def create(
        self,
        machine_id: str,
        name: str,
        category: str = "fan",
        description: str = "",
        operating_condition: str = "",
        selected_model_policy: str = "stetho_msae",
        owner_id: str = "usr_admin",
    ) -> MachineProfile:
        """Create a new physical machine profile with an owner."""
        if self.get_by_id(machine_id):
            raise StethoAPIException(
                f"Machine with ID '{machine_id}' already exists",
                code="MACHINE_EXISTS",
                status_code=409,
            )

        now = get_utc_now_iso()
        with self.db.transaction() as cursor:
            cursor.execute("""
            INSERT INTO machines (
                machine_id, name, category, description, operating_condition,
                selected_model_policy, enrollment_status, enrolled_recording_count,
                calibration_status, calibration_profile, calibration_version, owner_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'uncalibrated', 0, 'uncalibrated', NULL, '1.0.0', ?, ?, ?);
            """, (
                machine_id,
                name,
                category,
                description,
                operating_condition,
                selected_model_policy,
                owner_id,
                now,
                now,
            ))
            cursor.execute("""
            INSERT OR IGNORE INTO assets (
                asset_id, name, category, description, operating_condition,
                assigned_model, assigned_model_version, calibration_status,
                calibration_version, enrolled_recording_count, metadata_json,
                owner_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, '1.0.0', 'uncalibrated', '1.0.0', 0, '{}', ?, ?, ?);
            """, (
                machine_id,
                name,
                category,
                description,
                operating_condition,
                selected_model_policy,
                owner_id,
                now,
                now,
            ))

        return self.get_or_raise(machine_id)

    def increment_enrollment(
        self,
        machine_id: str,
        new_count: Optional[int] = None,
        new_status: Optional[str] = None,
    ) -> MachineProfile:
        """Increment or set enrolled count and compute appropriate enrollment/calibration status."""
        machine = self.get_or_raise(machine_id)
        count = new_count if new_count is not None else machine.enrolled_recording_count + 1
        status = new_status or ("calibrated" if count >= 30 else ("provisional" if count >= 1 else "uncalibrated"))
        profile_dict = machine.calibration_profile or {}
        return self.update_calibration_data(
            machine_id=machine_id,
            profile_dict=profile_dict,
            version=machine.calibration_version,
            status=status,
            enrolled_count=count,
        )

    def update_calibration_data(
        self,
        machine_id: str,
        profile_dict: Dict[str, Any],
        version: str,
        status: str,
        enrolled_count: int,
    ) -> MachineProfile:
        """Atomically persist newly recomputed calibration thresholds, version, and status."""
        now = get_utc_now_iso()
        profile_json = json.dumps(profile_dict)

        with self.db.transaction() as cursor:
            cursor.execute("""
            UPDATE machines
            SET calibration_profile = ?,
                calibration_version = ?,
                calibration_status = ?,
                enrollment_status = ?,
                enrolled_recording_count = ?,
                updated_at = ?
            WHERE machine_id = ?;
            """, (profile_json, version, status, status, enrolled_count, now, machine_id))

        return self.get_or_raise(machine_id)
