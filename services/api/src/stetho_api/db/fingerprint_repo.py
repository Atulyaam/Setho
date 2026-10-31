"""Repository layer for Machine Acoustic Fingerprints, QR Identity, and Drift Analytics.

Implements persistence and retrieval for:
- Machine Acoustic Fingerprints (Feature 1)
- Acoustic Drift Timelines (Feature 2)
- Fleet Inspection Priority (Feature 5)
- QR Identity Tokens (Feature 6)
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from stetho_api.db.database import DatabaseManager, get_utc_now_iso

logger = logging.getLogger("stetho.fingerprint_repo")


class FingerprintRepository:
    """Manages persistent acoustic fingerprints, QR tokens, and priority metrics."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_or_update_fingerprint(
        self,
        machine_id: str,
        spectral_centroid_hz: float,
        sub_bass_energy_db: float,
        low_energy_db: float,
        mid_energy_db: float,
        high_energy_db: float,
        baseline_reconstruction_error: float,
        baseline_quality_score: float,
        sample_count: int = 1,
        fingerprint_version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """Save or update persistent acoustic fingerprint reference for a machine."""
        now = get_utc_now_iso()
        with self.db.transaction() as cur:
            cur.execute("SELECT fingerprint_id FROM machine_fingerprints WHERE machine_id = ?;", (machine_id,))
            existing = cur.fetchone()

            if existing:
                fp_id = existing[0]
                cur.execute(
                    """
                    UPDATE machine_fingerprints
                    SET spectral_centroid_hz = ?,
                        sub_bass_energy_db = ?,
                        low_energy_db = ?,
                        mid_energy_db = ?,
                        high_energy_db = ?,
                        baseline_reconstruction_error = ?,
                        baseline_quality_score = ?,
                        sample_count = ?,
                        fingerprint_version = ?,
                        updated_at = ?
                    WHERE machine_id = ?;
                    """,
                    (
                        spectral_centroid_hz,
                        sub_bass_energy_db,
                        low_energy_db,
                        mid_energy_db,
                        high_energy_db,
                        baseline_reconstruction_error,
                        baseline_quality_score,
                        sample_count,
                        fingerprint_version,
                        now,
                        machine_id,
                    ),
                )
            else:
                fp_id = f"fp_{uuid.uuid4().hex[:12]}"
                cur.execute(
                    """
                    INSERT INTO machine_fingerprints (
                        fingerprint_id, machine_id, fingerprint_version,
                        spectral_centroid_hz, sub_bass_energy_db, low_energy_db,
                        mid_energy_db, high_energy_db, baseline_reconstruction_error,
                        baseline_quality_score, sample_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        fp_id,
                        machine_id,
                        fingerprint_version,
                        spectral_centroid_hz,
                        sub_bass_energy_db,
                        low_energy_db,
                        mid_energy_db,
                        high_energy_db,
                        baseline_reconstruction_error,
                        baseline_quality_score,
                        sample_count,
                        now,
                        now,
                    ),
                )

        return self.get_fingerprint(machine_id)  # type: ignore

    def get_fingerprint(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the reference acoustic fingerprint for a machine."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT fingerprint_id, machine_id, fingerprint_version,
                   spectral_centroid_hz, sub_bass_energy_db, low_energy_db,
                   mid_energy_db, high_energy_db, baseline_reconstruction_error,
                   baseline_quality_score, sample_count, created_at, updated_at
            FROM machine_fingerprints
            WHERE machine_id = ?;
            """,
            (machine_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def get_drift_timeline(self, machine_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve chronological inspection records formatted as a drift timeline."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT analysis_id, machine_id, model_name, model_version,
                   quality_status, decision, anomaly_score, operating_threshold,
                   is_provisional, processing_duration_ms, spectral_drift_pct,
                   dominant_band_deviation, maintenance_urgency, metrics, created_at
            FROM analyses
            WHERE machine_id = ?
            ORDER BY created_at ASC
            LIMIT ?;
            """,
            (machine_id, limit),
        )
        rows = cur.fetchall()
        timeline = []
        for r in rows:
            item = dict(r)
            if item.get("metrics"):
                try:
                    item["metrics"] = json.loads(item["metrics"])
                except Exception:
                    pass
            timeline.append(item)
        return timeline

    def get_or_create_qr_token(self, machine_id: str) -> Dict[str, Any]:
        """Retrieve active QR token or create a new cryptographically unique token."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT token_id, machine_id, qr_token, is_active, created_at, expires_at
            FROM machine_qr_tokens
            WHERE machine_id = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (machine_id,),
        )
        row = cur.fetchone()
        if row:
            return dict(row)

        # Create new token
        token_id = f"qrt_{uuid.uuid4().hex[:12]}"
        qr_token = f"stetho_qr_{uuid.uuid4().hex}"
        now = get_utc_now_iso()
        with self.db.transaction() as tx:
            tx.execute(
                """
                INSERT INTO machine_qr_tokens (token_id, machine_id, qr_token, is_active, created_at)
                VALUES (?, ?, ?, 1, ?);
                """,
                (token_id, machine_id, qr_token, now),
            )
        return {
            "token_id": token_id,
            "machine_id": machine_id,
            "qr_token": qr_token,
            "is_active": 1,
            "created_at": now,
            "expires_at": None,
        }

    def rotate_qr_token(self, machine_id: str) -> Dict[str, Any]:
        """Revoke existing QR tokens and issue a fresh token."""
        token_id = f"qrt_{uuid.uuid4().hex[:12]}"
        qr_token = f"stetho_qr_{uuid.uuid4().hex}"
        now = get_utc_now_iso()
        with self.db.transaction() as tx:
            tx.execute(
                "UPDATE machine_qr_tokens SET is_active = 0 WHERE machine_id = ?;",
                (machine_id,),
            )
            tx.execute(
                """
                INSERT INTO machine_qr_tokens (token_id, machine_id, qr_token, is_active, created_at)
                VALUES (?, ?, ?, 1, ?);
                """,
                (token_id, machine_id, qr_token, now),
            )
        return {
            "token_id": token_id,
            "machine_id": machine_id,
            "qr_token": qr_token,
            "is_active": 1,
            "created_at": now,
            "expires_at": None,
        }

    def get_machine_by_qr_token(self, qr_token: str) -> Optional[Dict[str, Any]]:
        """Resolve a QR token to its corresponding machine record."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.*, t.token_id, t.created_at as token_issued_at
            FROM machine_qr_tokens t
            JOIN machines m ON t.machine_id = m.machine_id
            WHERE t.qr_token = ? AND t.is_active = 1;
            """,
            (qr_token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def compute_fleet_priority(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compute transparent multi-factor fleet inspection priorities.

        Priority formula:
        Score = w_dev * max(0, (score - threshold) / threshold * 50)
              + w_unusual * (unusual_inspections_count * 15)
              + w_age * min(30, days_since_last_inspection * 2)
              + w_inconclusive * (inconclusive_count * 10)
        """
        conn = self.db.get_connection()
        cur = conn.cursor()

        query = "SELECT machine_id, name, category, enrollment_status, calibration_status, enrolled_recording_count, last_inspection_at FROM machines"
        params = ()
        if owner_id:
            query += " WHERE owner_id = ? OR owner_id = 'usr_admin'"
            params = (owner_id,)  # type: ignore

        cur.execute(query, params)
        machines = [dict(r) for r in cur.fetchall()]

        priority_list = []
        for m in machines:
            m_id = m["machine_id"]
            # Fetch recent analysis summary
            cur.execute(
                """
                SELECT decision, anomaly_score, operating_threshold, created_at
                FROM analyses
                WHERE machine_id = ?
                ORDER BY created_at DESC
                LIMIT 10;
                """,
                (m_id,),
            )
            recent_analyses = [dict(r) for r in cur.fetchall()]

            unusual_count = sum(1 for a in recent_analyses if a["decision"] == "UNUSUAL")
            inconclusive_count = sum(1 for a in recent_analyses if a["decision"] == "INCONCLUSIVE")
            normal_count = sum(1 for a in recent_analyses if a["decision"] == "NORMAL_LIKE")

            latest = recent_analyses[0] if recent_analyses else None
            score = 0.0
            reasons = []

            # 1. Acoustic deviation factor
            if latest and latest["anomaly_score"] is not None and latest["operating_threshold"]:
                thresh = latest["operating_threshold"]
                ascore = latest["anomaly_score"]
                if ascore > thresh:
                    dev_pct = ((ascore - thresh) / thresh) * 100.0
                    dev_points = min(40.0, dev_pct * 0.5)
                    score += dev_points
                    reasons.append(f"Acoustic deviation of {dev_pct:.1f}% above operating threshold")

            # 2. Repeated unusual results
            if unusual_count > 0:
                unusual_points = min(30.0, unusual_count * 15.0)
                score += unusual_points
                reasons.append(f"{unusual_count} unusual acoustic deviation(s) in recent inspections")

            # 3. Inconclusive inspections
            if inconclusive_count > 0:
                inconc_points = min(15.0, inconclusive_count * 7.5)
                score += inconc_points
                reasons.append(f"{inconclusive_count} inconclusive recording(s) requiring verification")

            # 4. Inspection age factor
            if latest:
                try:
                    last_dt = datetime.fromisoformat(latest["created_at"])
                    now_dt = datetime.now(timezone.utc)
                    days_ago = max(0, (now_dt - last_dt).days)
                    age_points = min(20.0, days_ago * 2.0)
                    score += age_points
                    if days_ago > 3:
                        reasons.append(f"Last inspected {days_ago} days ago")
                except Exception:
                    pass
            elif m["enrollment_status"] == "calibrated":
                score += 15.0
                reasons.append("Calibrated machine has no recorded baseline inspections")

            # Assign priority tier
            total_score = round(min(100.0, score), 1)
            if total_score >= 70.0:
                tier = "CRITICAL"
            elif total_score >= 45.0:
                tier = "HIGH"
            elif total_score >= 20.0:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            if not reasons:
                reasons.append("Routine acoustic status normal")

            priority_list.append({
                "machine_id": m["machine_id"],
                "name": m["name"],
                "category": m["category"],
                "calibration_status": m["calibration_status"],
                "priority_score": total_score,
                "priority_tier": tier,
                "reasons": reasons,
                "recent_inspections_count": len(recent_analyses),
                "unusual_count": unusual_count,
                "normal_count": normal_count,
                "latest_inspection_at": latest["created_at"] if latest else None,
                "latest_decision": latest["decision"] if latest else "UNINSPECTED",
            })

        # Sort descending by priority score
        priority_list.sort(key=lambda x: x["priority_score"], reverse=True)
        return priority_list
