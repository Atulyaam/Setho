"""Asset Repository for Stetho API (Sprint 7).

Manages persistent industrial machine asset records across categories (fan, pump, valve),
assigned model bindings, and calibration status with strict machine isolation.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional, Union
import uuid

from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import ConflictException, ResourceNotFoundException, ValidationException

logger = logging.getLogger("stetho.asset_repo")

VALID_CATEGORIES = {"fan", "pump", "valve"}
VALID_STATUSES = {"uncalibrated", "provisional", "calibrated"}


@dataclass
class Asset:
    """Represents a monitored physical industrial machine asset."""
    asset_id: str
    name: str
    category: str
    description: str = ""
    operating_condition: str = ""
    assigned_model: str = "cascade_ensemble"
    assigned_model_version: str = "1.0.0"
    calibration_status: str = "uncalibrated"
    calibration_version: str = "1.0.0"
    enrolled_recording_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    owner_id: str = "usr_admin"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "operating_condition": self.operating_condition,
            "assigned_model": self.assigned_model,
            "assigned_model_version": self.assigned_model_version,
            "calibration_status": self.calibration_status,
            "calibration_version": self.calibration_version,
            "enrolled_recording_count": self.enrolled_recording_count,
            "metadata": self.metadata,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AssetRepository:
    """Repository handling CRUD operations and queries for physical assets."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(
        self,
        asset_id: str,
        name: str,
        category: str,
        description: str = "",
        operating_condition: str = "",
        assigned_model: str = "cascade_ensemble",
        assigned_model_version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        owner_id: str = "usr_admin",
    ) -> Asset:
        """Create a new industrial asset profile."""
        cat_lower = category.lower().strip()
        if cat_lower not in VALID_CATEGORIES:
            raise ValidationException(
                f"Invalid asset category '{category}'. Must be one of: {sorted(list(VALID_CATEGORIES))}"
            )

        if not asset_id or not asset_id.strip():
            raise ValidationException("Asset ID cannot be empty.")
        if not name or not name.strip():
            raise ValidationException("Asset name cannot be empty.")

        existing = self.get(asset_id)
        if existing:
            raise ConflictException(f"Asset with ID '{asset_id}' already exists.")

        now = get_utc_now_iso()
        meta_dict = metadata or {}
        meta_json = json.dumps(meta_dict)

        with self.db.transaction() as cur:
            cur.execute("""
            INSERT INTO assets (
                asset_id, name, category, description, operating_condition,
                assigned_model, assigned_model_version, calibration_status,
                calibration_version, enrolled_recording_count, metadata_json,
                owner_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'uncalibrated', '1.0.0', 0, ?, ?, ?, ?);
            """, (
                asset_id, name, cat_lower, description, operating_condition,
                assigned_model, assigned_model_version, meta_json,
                owner_id, now, now
            ))

            # Maintain backward compatibility with machines table
            cur.execute("""
            INSERT OR IGNORE INTO machines (
                machine_id, name, category, description, operating_condition,
                selected_model_policy, enrollment_status, enrolled_recording_count,
                calibration_status, calibration_version, owner_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'uncalibrated', 0, 'uncalibrated', '1.0.0', ?, ?, ?);
            """, (
                asset_id, name, cat_lower, description, operating_condition,
                assigned_model, owner_id, now, now
            ))

        logger.info(f"Created asset '{asset_id}' ({name}, category={cat_lower})")
        return self.get_or_raise(asset_id)

    def get(self, asset_id: str) -> Optional[Asset]:
        """Retrieve an asset by ID or return None."""
        conn = self.db.get_connection()
        row = conn.execute("SELECT * FROM assets WHERE asset_id = ?;", (asset_id,)).fetchone()
        if not row:
            # Fallback to machines table for transparent backward compatibility
            m_row = conn.execute("SELECT * FROM machines WHERE machine_id = ?;", (asset_id,)).fetchone()
            if m_row:
                try:
                    with self.db.transaction() as cur:
                        cur.execute("""
                        INSERT OR IGNORE INTO assets (
                            asset_id, name, category, description, operating_condition,
                            assigned_model, assigned_model_version, calibration_status,
                            calibration_version, enrolled_recording_count, metadata_json,
                            owner_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, '1.0.0', ?, ?, ?, '{}', ?, ?, ?);
                        """, (
                            m_row["machine_id"], m_row["name"], m_row["category"],
                            m_row["description"] or "", m_row["operating_condition"] or "",
                            m_row["selected_model_policy"], m_row["calibration_status"] or "uncalibrated",
                            m_row["calibration_version"] or "1.0.0", m_row["enrolled_recording_count"] or 0,
                            m_row["owner_id"] or "usr_admin", m_row["created_at"], m_row["updated_at"]
                        ))
                except Exception:
                    pass
                return Asset(
                    asset_id=m_row["machine_id"],
                    name=m_row["name"],
                    category=m_row["category"],
                    description=m_row["description"] or "",
                    operating_condition=m_row["operating_condition"] or "",
                    assigned_model=m_row["selected_model_policy"],
                    assigned_model_version="1.0.0",
                    calibration_status=m_row["calibration_status"],
                    calibration_version=m_row["calibration_version"] or "1.0.0",
                    enrolled_recording_count=m_row["enrolled_recording_count"] or 0,
                    metadata={},
                    owner_id=m_row["owner_id"] or "usr_admin",
                    created_at=m_row["created_at"],
                    updated_at=m_row["updated_at"],
                )
            return None

        try:
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except Exception:
            meta = {}

        return Asset(
            asset_id=row["asset_id"],
            name=row["name"],
            category=row["category"],
            description=row["description"] or "",
            operating_condition=row["operating_condition"] or "",
            assigned_model=row["assigned_model"],
            assigned_model_version=row["assigned_model_version"],
            calibration_status=row["calibration_status"],
            calibration_version=row["calibration_version"],
            enrolled_recording_count=row["enrolled_recording_count"],
            metadata=meta,
            owner_id=row["owner_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_or_raise(self, asset_id: str) -> Asset:
        """Retrieve an asset by ID or raise ResourceNotFoundException."""
        asset = self.get(asset_id)
        if not asset:
            raise ResourceNotFoundException("Asset", asset_id)
        return asset

    def list_all(
        self,
        user_id: Optional[str] = None,
        is_admin: bool = False,
        category: Optional[str] = None,
    ) -> List[Asset]:
        """List all assets accessible to the user, optionally filtered by category."""
        conn = self.db.get_connection()
        query = "SELECT * FROM assets WHERE 1=1"
        params: List[Any] = []

        if not is_admin and user_id:
            query += " AND (owner_id = ? OR owner_id = 'usr_admin')"
            params.append(user_id)

        if category:
            query += " AND category = ?"
            params.append(category.lower().strip())

        query += " ORDER BY created_at DESC;"
        rows = conn.execute(query, params).fetchall()

        results = []
        for r in rows:
            try:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                meta = {}
            results.append(
                Asset(
                    asset_id=r["asset_id"],
                    name=r["name"],
                    category=r["category"],
                    description=r["description"] or "",
                    operating_condition=r["operating_condition"] or "",
                    assigned_model=r["assigned_model"],
                    assigned_model_version=r["assigned_model_version"],
                    calibration_status=r["calibration_status"],
                    calibration_version=r["calibration_version"],
                    enrolled_recording_count=r["enrolled_recording_count"],
                    metadata=meta,
                    owner_id=r["owner_id"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            )
        return results

    def update_calibration_status(
        self,
        asset_id: str,
        calibration_status: str,
        calibration_version: str,
        enrolled_count: int,
    ) -> None:
        """Update calibration status and version atomically."""
        if calibration_status not in VALID_STATUSES:
            raise ValidationException(f"Invalid calibration status '{calibration_status}'")

        now = get_utc_now_iso()
        with self.db.transaction() as cur:
            cur.execute("""
            UPDATE assets
            SET calibration_status = ?,
                calibration_version = ?,
                enrolled_recording_count = ?,
                updated_at = ?
            WHERE asset_id = ?;
            """, (calibration_status, calibration_version, enrolled_count, now, asset_id))

            # Keep machines table synchronized
            cur.execute("""
            UPDATE machines
            SET calibration_status = ?,
                calibration_version = ?,
                enrolled_recording_count = ?,
                updated_at = ?
            WHERE machine_id = ?;
            """, (calibration_status, calibration_version, enrolled_count, now, asset_id))

        logger.info(
            f"Updated calibration for asset '{asset_id}': status={calibration_status}, "
            f"ver={calibration_version}, count={enrolled_count}"
        )

    def update_assigned_model(
        self,
        asset_id: str,
        assigned_model: str,
        assigned_model_version: str = "1.0.0",
    ) -> Asset:
        """Update assigned model policy for an asset."""
        now = get_utc_now_iso()
        with self.db.transaction() as cur:
            cur.execute("""
            UPDATE assets
            SET assigned_model = ?,
                assigned_model_version = ?,
                updated_at = ?
            WHERE asset_id = ?;
            """, (assigned_model, assigned_model_version, now, asset_id))

            cur.execute("""
            UPDATE machines
            SET selected_model_policy = ?,
                updated_at = ?
            WHERE machine_id = ?;
            """, (assigned_model, now, asset_id))

        return self.get_or_raise(asset_id)
