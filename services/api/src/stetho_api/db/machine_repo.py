"""Machine Profile Repository for Stetho API."""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional

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
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_or_raise(self, machine_id: str) -> MachineProfile:
        """Fetch machine profile or raise ResourceNotFoundException."""
        machine = self.get_by_id(machine_id)
        if not machine:
            raise ResourceNotFoundException("Machine", machine_id)
        return machine

    def list_all(self) -> List[MachineProfile]:
        """List all registered machine profiles."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM machines ORDER BY machine_id ASC;")
        rows = cursor.fetchall()
        return [
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
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def create(
        self,
        machine_id: str,
        name: str,
        category: str = "fan",
        description: str = "",
        operating_condition: str = "",
        selected_model_policy: str = "stetho_msae",
    ) -> MachineProfile:
        """Create a new physical machine profile."""
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
                calibration_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'uncalibrated', 0, 'uncalibrated', ?, ?);
            """, (
                machine_id,
                name,
                category,
                description,
                operating_condition,
                selected_model_policy,
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
        """Increment enrolled recording count and update calibration status."""
        machine = self.get_or_raise(machine_id)
        count = new_count if new_count is not None else machine.enrolled_recording_count + 1

        # Calibration rules from Sprint 3.5:
        # N < 30: provisional calibration
        # N >= 30: fully calibrated
        if new_status:
            status = new_status
        elif count >= 30:
            status = "calibrated"
        elif count >= 1:
            status = "provisional"
        else:
            status = "uncalibrated"

        now = get_utc_now_iso()
        with self.db.transaction() as cursor:
            cursor.execute("""
            UPDATE machines
            SET enrolled_recording_count = ?,
                enrollment_status = ?,
                calibration_status = ?,
                updated_at = ?
            WHERE machine_id = ?;
            """, (count, status, status, now, machine_id))

        return self.get_or_raise(machine_id)

