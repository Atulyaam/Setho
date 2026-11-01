"""Calibration Repository for Stetho API (Sprint 7).

Manages persistent few-shot calibration profiles, latent space centering statistics,
audit history, and version rollbacks with strict machine-level isolation.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import ResourceNotFoundException, ValidationException

logger = logging.getLogger("stetho.calibration_repo")


@dataclass
class CalibrationProfile:
    """Persistent machine-specific statistical calibration profile."""
    profile_id: str
    asset_id: str
    model_version: str
    calibration_version: str
    operating_threshold: float
    theta_lower: float
    theta_upper: float
    baseline_mean: float
    baseline_std: float
    enrolled_count: int
    latent_mean: Optional[List[float]] = None
    latent_cov: Optional[List[float]] = None
    scores: List[float] = field(default_factory=list)
    is_active: bool = True
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "asset_id": self.asset_id,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "operating_threshold": round(self.operating_threshold, 6),
            "theta_lower": round(self.theta_lower, 6),
            "theta_upper": round(self.theta_upper, 6),
            "baseline_mean": round(self.baseline_mean, 6),
            "baseline_std": round(self.baseline_std, 6),
            "enrolled_count": self.enrolled_count,
            "latent_mean": self.latent_mean,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }


@dataclass
class CalibrationHistoryRecord:
    """Audit record capturing calibration lifecycle events."""
    history_id: str
    asset_id: str
    event_type: str  # "CALIBRATION", "RECALIBRATION", "ROLLBACK"
    previous_version: Optional[str]
    new_version: str
    previous_threshold: Optional[float]
    new_threshold: float
    sample_count: int
    operator_id: str
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history_id": self.history_id,
            "asset_id": self.asset_id,
            "event_type": self.event_type,
            "previous_version": self.previous_version,
            "new_version": self.new_version,
            "previous_threshold": round(self.previous_threshold, 6) if self.previous_threshold is not None else None,
            "new_threshold": round(self.new_threshold, 6),
            "sample_count": self.sample_count,
            "operator_id": self.operator_id,
            "notes": self.notes,
            "created_at": self.created_at,
        }


class CalibrationRepository:
    """Repository handling calibration profile storage, history, and rollbacks."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_profile(self, profile: CalibrationProfile) -> CalibrationProfile:
        """Save a new calibration profile and mark previous active profiles for this asset as inactive."""
        now = get_utc_now_iso()
        if not profile.profile_id:
            profile.profile_id = f"cal_{uuid.uuid4().hex[:12]}"
        if not profile.created_at:
            profile.created_at = now

        latent_mean_json = json.dumps(profile.latent_mean) if profile.latent_mean else None
        latent_cov_json = json.dumps(profile.latent_cov) if profile.latent_cov else None
        scores_json = json.dumps(profile.scores)

        with self.db.transaction() as cur:
            # Deactivate existing active profiles for this asset
            cur.execute("""
            UPDATE calibration_profiles
            SET is_active = 0
            WHERE asset_id = ?;
            """, (profile.asset_id,))

            # Insert new active profile
            cur.execute("""
            INSERT INTO calibration_profiles (
                profile_id, asset_id, model_version, calibration_version,
                operating_threshold, theta_lower, theta_upper, baseline_mean,
                baseline_std, enrolled_count, latent_mean_json, latent_cov_json,
                scores_json, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?);
            """, (
                profile.profile_id, profile.asset_id, profile.model_version,
                profile.calibration_version, profile.operating_threshold,
                profile.theta_lower, profile.theta_upper, profile.baseline_mean,
                profile.baseline_std, profile.enrolled_count, latent_mean_json,
                latent_cov_json, scores_json, profile.created_at
            ))

        logger.info(
            f"Saved calibration profile '{profile.profile_id}' for asset '{profile.asset_id}' "
            f"(ver={profile.calibration_version}, threshold={profile.operating_threshold:.5f})"
        )
        return profile

    def get_active_profile(self, asset_id: str) -> Optional[CalibrationProfile]:
        """Retrieve the currently active calibration profile for an asset."""
        conn = self.db.get_connection()
        row = conn.execute("""
        SELECT * FROM calibration_profiles
        WHERE asset_id = ? AND is_active = 1
        ORDER BY created_at DESC LIMIT 1;
        """, (asset_id,)).fetchone()

        if not row:
            return None

        return self._row_to_profile(row)

    def list_profiles(self, asset_id: str) -> List[CalibrationProfile]:
        """List all historical calibration profiles for an asset in chronological order."""
        conn = self.db.get_connection()
        rows = conn.execute("""
        SELECT * FROM calibration_profiles
        WHERE asset_id = ?
        ORDER BY created_at DESC;
        """, (asset_id,)).fetchall()

        return [self._row_to_profile(r) for r in rows]

    def add_history(self, record: CalibrationHistoryRecord) -> CalibrationHistoryRecord:
        """Add an audit entry to calibration history."""
        now = get_utc_now_iso()
        if not record.history_id:
            record.history_id = f"calhist_{uuid.uuid4().hex[:12]}"
        if not record.created_at:
            record.created_at = now

        with self.db.transaction() as cur:
            cur.execute("""
            INSERT INTO calibration_history (
                history_id, asset_id, event_type, previous_version, new_version,
                previous_threshold, new_threshold, sample_count, operator_id,
                notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.history_id, record.asset_id, record.event_type,
                record.previous_version, record.new_version,
                record.previous_threshold, record.new_threshold,
                record.sample_count, record.operator_id, record.notes,
                record.created_at
            ))

        logger.info(
            f"Added calibration history for asset '{record.asset_id}': {record.event_type} "
            f"({record.previous_version} -> {record.new_version})"
        )
        return record

    def list_history(self, asset_id: str) -> List[CalibrationHistoryRecord]:
        """List calibration audit history for an asset."""
        conn = self.db.get_connection()
        rows = conn.execute("""
        SELECT * FROM calibration_history
        WHERE asset_id = ?
        ORDER BY created_at DESC;
        """, (asset_id,)).fetchall()

        results = []
        for r in rows:
            results.append(
                CalibrationHistoryRecord(
                    history_id=r["history_id"],
                    asset_id=r["asset_id"],
                    event_type=r["event_type"],
                    previous_version=r["previous_version"],
                    new_version=r["new_version"],
                    previous_threshold=r["previous_threshold"],
                    new_threshold=r["new_threshold"],
                    sample_count=r["sample_count"],
                    operator_id=r["operator_id"],
                    notes=r["notes"] or "",
                    created_at=r["created_at"],
                )
            )
        return results

    def rollback(
        self,
        asset_id: str,
        target_version: Optional[str] = None,
        operator_id: str = "usr_admin",
    ) -> CalibrationProfile:
        """Rollback calibration to a prior version (or the immediate previous version if none specified)."""
        profiles = self.list_profiles(asset_id)
        if len(profiles) < 2 and not target_version:
            raise ValidationException("Cannot rollback: No previous calibration profile exists for this asset.")

        current_active = next((p for p in profiles if p.is_active), None)

        target_profile: Optional[CalibrationProfile] = None
        if target_version:
            target_profile = next((p for p in profiles if p.calibration_version == target_version), None)
            if not target_profile:
                raise ResourceNotFoundException(
                    "CalibrationProfile", f"Version '{target_version}' for asset '{asset_id}'"
                )
        else:
            # Find the first inactive profile (immediate predecessor)
            target_profile = next((p for p in profiles if not p.is_active), None)
            if not target_profile:
                raise ValidationException("No prior calibration profile available for rollback.")

        with self.db.transaction() as cur:
            # Deactivate all
            cur.execute("UPDATE calibration_profiles SET is_active = 0 WHERE asset_id = ?;", (asset_id,))
            # Reactivate target
            cur.execute("""
            UPDATE calibration_profiles SET is_active = 1 WHERE profile_id = ?;
            """, (target_profile.profile_id,))

        target_profile.is_active = True

        # Log rollback event
        hist = CalibrationHistoryRecord(
            history_id=f"calhist_{uuid.uuid4().hex[:12]}",
            asset_id=asset_id,
            event_type="ROLLBACK",
            previous_version=current_active.calibration_version if current_active else None,
            new_version=target_profile.calibration_version,
            previous_threshold=current_active.operating_threshold if current_active else None,
            new_threshold=target_profile.operating_threshold,
            sample_count=target_profile.enrolled_count,
            operator_id=operator_id,
            notes=f"Rolled back to calibration profile {target_profile.calibration_version}",
        )
        self.add_history(hist)

        logger.info(
            f"Successfully rolled back asset '{asset_id}' to calibration version '{target_profile.calibration_version}'"
        )
        return target_profile

    def _row_to_profile(self, row: Any) -> CalibrationProfile:
        """Deserialize database row to CalibrationProfile."""
        latent_m = json.loads(row["latent_mean_json"]) if row["latent_mean_json"] else None
        latent_c = json.loads(row["latent_cov_json"]) if row["latent_cov_json"] else None
        try:
            scores = json.loads(row["scores_json"]) if row["scores_json"] else []
        except Exception:
            scores = []

        return CalibrationProfile(
            profile_id=row["profile_id"],
            asset_id=row["asset_id"],
            model_version=row["model_version"],
            calibration_version=row["calibration_version"],
            operating_threshold=row["operating_threshold"],
            theta_lower=row["theta_lower"],
            theta_upper=row["theta_upper"],
            baseline_mean=row["baseline_mean"],
            baseline_std=row["baseline_std"],
            enrolled_count=row["enrolled_count"],
            latent_mean=latent_m,
            latent_cov=latent_c,
            scores=scores,
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
        )
