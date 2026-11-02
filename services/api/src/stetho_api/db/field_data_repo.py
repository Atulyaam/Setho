"""Field Data Collection, Annotation, and Dataset Governance Repository.

Provides thread-safe persistence and business rule enforcement for:
- Importing field audio recordings with hash deduplication and quality validation
- Multi-revision ground-truth annotations with evidence tracking
- Dataset role separation and immutable sealed evaluation manifests
- Live dashboard telemetry and data collection protocol export
"""

from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import soundfile as sf

from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import StethoAPIException

logger = logging.getLogger("stetho.field_data_repo")


class DuplicateRecordingException(StethoAPIException):
    """Raised when an uploaded recording has a duplicate SHA-256 hash."""
    def __init__(self, message: str, existing_id: Optional[str] = None):
        super().__init__(message=message, status_code=409, code="DUPLICATE_RECORDING")
        self.existing_id = existing_id


class InvalidAudioQualityException(StethoAPIException):
    """Raised when an uploaded audio recording violates quality constraints."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=422, code="INVALID_AUDIO_QUALITY", details=details)


class DatasetRoleConflictException(StethoAPIException):
    """Raised when a dataset role transition violates governance boundaries."""
    def __init__(self, message: str):
        super().__init__(message=message, status_code=422, code="DATASET_ROLE_CONFLICT")


class ManifestSealedException(StethoAPIException):
    """Raised when an attempt is made to modify an immutable sealed manifest."""
    def __init__(self, message: str, manifest_id: Optional[str] = None):
        super().__init__(message=message, status_code=409, code="MANIFEST_ALREADY_SEALED")
        self.manifest_id = manifest_id


class FieldDataRepo:
    """Repository managing field recordings, annotations, and dataset manifests."""

    def __init__(self, db: DatabaseManager, storage_base_dir: Optional[Union[str, Path]] = None):
        self.db = db
        self.storage_base_dir = Path(storage_base_dir or "./data/storage/field_recordings")
        self.storage_base_dir.mkdir(parents=True, exist_ok=True)

    def import_recording(
        self,
        file_bytes: bytes,
        original_filename: str,
        machine_category: str,
        machine_id: str,
        sensor_id: str,
        operator_id: str,
        channel_index: int = 0,
        microphone_position: str = "primary_bearing",
        operating_condition: str = "nominal",
        load_level: str = "full_load",
        captured_at: Optional[str] = None,
        asset_id: Optional[str] = None,
        provenance_source: str = "field_pilot",
        consent_status: str = "authorized",
        dataset_role: str = "unassigned",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a field WAV recording with cryptographic hash check and quality gating.

        Args:
            file_bytes: Raw binary audio payload
            original_filename: Client filename
            machine_category: Equipment category (fan, pump, valve, etc.)
            machine_id: Machine identifier
            sensor_id: Transducer/microphone serial number
            operator_id: Technician user ID
            channel_index: Monitored capsule index
            microphone_position: Physical placement description
            operating_condition: Machine operating state
            load_level: Machine mechanical load
            captured_at: ISO timestamp of field capture
            asset_id: Optional asset linkage
            provenance_source: Origin of recording
            consent_status: Authorization status
            metadata: Additional operational metadata

        Returns:
            Dict representing the created field_recording record.
        """
        now = get_utc_now_iso()
        captured_timestamp = captured_at or now

        # Role check: Cannot import directly into calibration_enrollment
        if dataset_role == "calibration_enrollment":
            raise DatasetRoleConflictException(
                "Cannot import directly into 'calibration_enrollment'. "
                "Recordings must first be ingested and verified as normal by an authorized operator."
            )

        # Path traversal guard on machine_category
        clean_category = re.sub(r"[^a-zA-Z0-9_\-]", "", machine_category).strip()
        if not clean_category or clean_category != machine_category:
            raise StethoAPIException(
                f"Invalid machine_category '{machine_category}'. Must contain only alphanumeric, underscore, or hyphen characters.",
                status_code=400,
                code="INVALID_CATEGORY_NAME",
            )
        dest_dir = (self.storage_base_dir / clean_category).resolve()
        if not str(dest_dir).startswith(str(self.storage_base_dir.resolve())):
            raise StethoAPIException(
                "Path traversal detected in destination directory.",
                status_code=400,
                code="PATH_TRAVERSAL_DETECTED",
            )
        clean_filename = re.sub(r"[\x00-\x1f\x7f]", "", Path(original_filename).name)[:255] or "recording.wav"

        # 1. Compute SHA-256 hash immediately
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        # 2. Duplicate Detection
        existing_id = None
        with self.db.transaction() as cur:
            cur.execute("SELECT recording_id FROM field_recordings WHERE sha256_hash = ?;", (sha256_hash,))
            existing = cur.fetchone()
            if existing:
                existing_id = existing["recording_id"]
                import_id = f"imp_{uuid.uuid4().hex[:12]}"
                cur.execute("""
                INSERT INTO field_import_history (
                    import_id, original_filename, sha256_hash, status,
                    recording_id, rejection_reason, operator_id, timestamp
                ) VALUES (?, ?, ?, 'DUPLICATE_REJECTED', ?, ?, ?, ?);
                """, (
                    import_id, clean_filename, sha256_hash,
                    existing_id, f"Duplicate file with identical hash already imported as {existing_id}",
                    operator_id, now
                ))

        if existing_id is not None:
            raise DuplicateRecordingException(
                f"Recording with hash {sha256_hash[:12]} already exists as {existing_id}.",
                existing_id=existing_id,
            )

        # 3. Format and Audio Validation
        try:
            with sf.SoundFile(io.BytesIO(file_bytes)) as sound_file:
                sample_rate = sound_file.samplerate
                channels = sound_file.channels
                frames = sound_file.frames
                duration_sec = float(frames) / float(sample_rate) if sample_rate > 0 else 0.0
                raw_audio = sound_file.read(dtype="float32")
        except Exception as e:
            with self.db.transaction() as cur:
                cur.execute("""
                INSERT INTO field_import_history (
                    import_id, original_filename, sha256_hash, status,
                    rejection_reason, operator_id, timestamp
                ) VALUES (?, ?, ?, 'VALIDATION_FAILED', ?, ?, ?);
                """, (
                    f"imp_{uuid.uuid4().hex[:12]}", clean_filename, sha256_hash,
                    f"Corrupt or unreadable audio format: {str(e)}", operator_id, now
                ))
            raise InvalidAudioQualityException(f"Unsupported or corrupt audio file: {str(e)}")

        # 4. Gating Rules
        rejection_reasons = []
        if sample_rate < 16000:
            rejection_reasons.append(
                f"SAMPLE_RATE_UNSUPPORTED: Sample rate {sample_rate} Hz is below minimum requirement of 16000 Hz."
            )
        if sample_rate > 96000:
            rejection_reasons.append(
                f"SAMPLE_RATE_EXCESSIVE: Sample rate {sample_rate} Hz exceeds maximum limit of 96000 Hz."
            )
        if duration_sec < 1.0:
            rejection_reasons.append(
                f"DURATION_TOO_SHORT: Duration {duration_sec:.2f}s is less than minimum 1.0s."
            )
        if duration_sec > 60.0:
            rejection_reasons.append(
                f"DURATION_TOO_LONG: Duration {duration_sec:.2f}s exceeds maximum allowed limit of 60.0s."
            )
        if channel_index >= channels:
            rejection_reasons.append(
                f"CHANNEL_OUT_OF_BOUNDS: Requested channel {channel_index} >= available {channels} channels."
            )

        # Check clipping and silence on target channel
        target_audio = raw_audio[:, channel_index] if raw_audio.ndim > 1 else raw_audio
        rms_linear = float(np.sqrt(np.mean(target_audio**2)))
        rms_dbfs = float(20.0 * np.log10(rms_linear + 1e-12))
        peak_amp = float(np.max(np.abs(target_audio)))
        clipping_ratio = float(np.mean(np.abs(target_audio) >= 0.999))

        if rms_dbfs < -60.0 or peak_amp < 1e-3:
            rejection_reasons.append(
                f"SIGNAL_SILENCE_OR_EMPTY: RMS {rms_dbfs:.1f} dBFS or peak {peak_amp:.4f} is empty/silence."
            )
        if clipping_ratio > 0.001:
            rejection_reasons.append(
                f"EXCESSIVE_DIGITAL_CLIPPING: Clipping ratio {clipping_ratio * 100:.2f}% exceeds 0.1% limit."
            )

        if rejection_reasons:
            combined_reason = " | ".join(rejection_reasons)
            with self.db.transaction() as cur:
                cur.execute("""
                INSERT INTO field_import_history (
                    import_id, original_filename, sha256_hash, status,
                    rejection_reason, operator_id, timestamp
                ) VALUES (?, ?, ?, 'VALIDATION_FAILED', ?, ?, ?);
                """, (
                    f"imp_{uuid.uuid4().hex[:12]}", clean_filename, sha256_hash,
                    combined_reason, operator_id, now
                ))
            raise InvalidAudioQualityException(combined_reason)

        # 5. Quality Telemetry
        quality_telemetry = {
            "rms_dbfs": round(rms_dbfs, 2),
            "peak_amplitude": round(peak_amp, 4),
            "clipping_ratio": round(clipping_ratio, 6),
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_sec": round(duration_sec, 2),
        }

        # 6. Save original audio file to disk
        recording_id = f"fld_rec_{uuid.uuid4().hex[:12]}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_path = dest_dir / f"{recording_id}.wav"

        with open(file_path, "wb") as f_out:
            f_out.write(file_bytes)

        # 7. Insert recording record and import history
        import_id = f"imp_{uuid.uuid4().hex[:12]}"
        meta_dict = metadata or {}

        with self.db.transaction() as cur:
            cur.execute("""
            INSERT INTO field_recordings (
                recording_id, asset_id, machine_category, machine_id, sensor_id,
                channel_index, microphone_position, sample_rate, channels,
                duration_sec, file_path, original_filename, sha256_hash,
                captured_at, operating_condition, load_level, operator_id,
                provenance_source, consent_status, dataset_role, current_label,
                verification_status, quality_status, quality_telemetry_json,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                recording_id, asset_id, clean_category, machine_id, sensor_id,
                channel_index, microphone_position, sample_rate, channels,
                duration_sec, str(file_path).replace("\\", "/"), clean_filename, sha256_hash,
                captured_timestamp, operating_condition, load_level, operator_id,
                provenance_source, consent_status, dataset_role, "unlabelled",
                "PENDING", "PASSED", json.dumps(quality_telemetry),
                json.dumps(meta_dict), now, now
            ))

            cur.execute("""
            INSERT INTO field_import_history (
                import_id, original_filename, sha256_hash, status,
                recording_id, rejection_reason, operator_id, timestamp
            ) VALUES (?, ?, ?, 'SUCCESS', ?, NULL, ?, ?);
            """, (
                import_id, clean_filename, sha256_hash,
                recording_id, operator_id, now
            ))

        return self.get_recording(recording_id)

    def get_recording(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single field recording by ID with its parsed telemetry and metadata."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM field_recordings WHERE recording_id = ?;", (recording_id,))
        row = cur.fetchone()
        if not row:
            return None
        rec = dict(row)
        rec["quality_telemetry"] = json.loads(rec.get("quality_telemetry_json") or "{}")
        rec["metadata"] = json.loads(rec.get("metadata_json") or "{}")
        return rec

    def list_recordings(
        self,
        machine_category: Optional[str] = None,
        machine_id: Optional[str] = None,
        dataset_role: Optional[str] = None,
        current_label: Optional[str] = None,
        verification_status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List field recordings with optional filtering and pagination."""
        conn = self.db.get_connection()
        cur = conn.cursor()

        query = "SELECT * FROM field_recordings WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM field_recordings WHERE 1=1"
        params: List[Any] = []

        if machine_category:
            query += " AND machine_category = ?"
            count_query += " AND machine_category = ?"
            params.append(machine_category)
        if machine_id:
            query += " AND machine_id = ?"
            count_query += " AND machine_id = ?"
            params.append(machine_id)
        if dataset_role:
            query += " AND dataset_role = ?"
            count_query += " AND dataset_role = ?"
            params.append(dataset_role)
        if current_label:
            query += " AND current_label = ?"
            count_query += " AND current_label = ?"
            params.append(current_label)
        if verification_status:
            query += " AND verification_status = ?"
            count_query += " AND verification_status = ?"
            params.append(verification_status)

        cur.execute(count_query, params)
        total_count = cur.fetchone()[0]

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?;"
        cur.execute(query, params + [limit, offset])
        rows = cur.fetchall()

        results = []
        for r in rows:
            rec = dict(r)
            rec["quality_telemetry"] = json.loads(rec.get("quality_telemetry_json") or "{}")
            rec["metadata"] = json.loads(rec.get("metadata_json") or "{}")
            results.append(rec)

        return results, total_count

    def add_annotation(
        self,
        recording_id: str,
        label: str,
        reason: str,
        evidence_source: str,
        annotator_id: str,
        inspection_notes: str = "",
    ) -> Dict[str, Any]:
        """Record a human ground-truth annotation with full revision history.

        Args:
            recording_id: Target recording
            label: 'normal', 'abnormal', 'uncertain', 'unlabelled'
            reason: Physical diagnostic justification
            evidence_source: Vibration analysis, thermal scan, teardown, etc.
            annotator_id: Operator/engineer user ID
            inspection_notes: Optional technician observations

        Returns:
            Dict representing the created annotation record.
        """
        valid_labels = {"normal", "abnormal", "uncertain", "unlabelled"}
        if label not in valid_labels:
            raise StethoAPIException(f"Invalid label '{label}'. Must be one of {valid_labels}", status_code=400)

        valid_evidence = {
            "VIBRATION_ORDER_ANALYSIS",
            "THERMAL_SCAN",
            "OIL_ANALYSIS",
            "TEARDOWN_INSPECTION",
            "VISUAL_INSPECTION",
            "OPERATIONAL_LOG",
            "OTHER_GROUND_TRUTH",
        }
        if evidence_source not in valid_evidence:
            raise StethoAPIException(
                f"Invalid evidence_source '{evidence_source}'. Must be one of {valid_evidence}",
                status_code=400,
            )

        now = get_utc_now_iso()

        with self.db.transaction() as cur:
            # 1. Verify recording exists
            cur.execute("SELECT recording_id, dataset_role, current_label FROM field_recordings WHERE recording_id = ?;", (recording_id,))
            rec = cur.fetchone()
            if not rec:
                raise StethoAPIException(f"Recording '{recording_id}' not found.", status_code=404)

            # 2. Enforce dataset role restriction: calibration cannot contain abnormal recordings
            if rec["dataset_role"] == "calibration_enrollment" and label == "abnormal":
                raise DatasetRoleConflictException(
                    "Calibration enrollment recordings cannot be annotated as abnormal. "
                    "Calibration data must represent confirmed healthy baseline operation."
                )

            # 2b. If recording belongs to a sealed evaluation manifest, prevent modifying ground-truth label!
            if rec["dataset_role"] == "sealed_evaluation":
                cur.execute("""
                SELECT manifest_id, name, recordings_json FROM dataset_manifests
                WHERE dataset_role = 'sealed_evaluation' AND is_sealed = 1;
                """)
                for manifest in cur.fetchall():
                    member_ids = json.loads(manifest["recordings_json"] or "[]")
                    if recording_id in member_ids:
                        raise ManifestSealedException(
                            f"Recording '{recording_id}' belongs to sealed evaluation manifest '{manifest['name']}' "
                            f"({manifest['manifest_id']}) and cannot have its ground-truth modified.",
                            manifest_id=manifest["manifest_id"],
                        )

            # 3. Supersede previous annotations
            cur.execute("""
            UPDATE recording_annotations
            SET verification_status = 'SUPERSEDED'
            WHERE recording_id = ? AND verification_status = 'VERIFIED';
            """, (recording_id,))

            # 4. Count existing annotations for revision number
            cur.execute("SELECT COUNT(*) FROM recording_annotations WHERE recording_id = ?;", (recording_id,))
            rev_num = cur.fetchone()[0] + 1

            annotation_id = f"ann_{uuid.uuid4().hex[:12]}"
            cur.execute("""
            INSERT INTO recording_annotations (
                annotation_id, recording_id, label, annotator_id, reason,
                evidence_source, verification_status, inspection_notes,
                revision_number, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'VERIFIED', ?, ?, ?);
            """, (
                annotation_id, recording_id, label, annotator_id, reason,
                evidence_source, inspection_notes, rev_num, now
            ))

            # 5. Update field_recordings current label & verification status
            v_status = "VERIFIED" if label in ["normal", "abnormal"] else "PENDING"
            cur.execute("""
            UPDATE field_recordings
            SET current_label = ?, verification_status = ?, updated_at = ?
            WHERE recording_id = ?;
            """, (label, v_status, now, recording_id))

        return {
            "annotation_id": annotation_id,
            "recording_id": recording_id,
            "label": label,
            "annotator_id": annotator_id,
            "reason": reason,
            "evidence_source": evidence_source,
            "verification_status": v_status,
            "inspection_notes": inspection_notes,
            "revision_number": rev_num,
            "created_at": now,
        }

    def get_annotation_history(self, recording_id: str) -> List[Dict[str, Any]]:
        """Retrieve audit history of all annotations for a recording."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute("""
        SELECT * FROM recording_annotations
        WHERE recording_id = ?
        ORDER BY revision_number ASC;
        """, (recording_id,))
        return [dict(r) for r in cur.fetchall()]

    def assign_dataset_role(self, recording_id: str, new_role: str, operator_id: str) -> Dict[str, Any]:
        """Assign or reclassify a recording's dataset role with boundary checks.

        Roles:
        - 'unassigned'
        - 'training'
        - 'development'
        - 'calibration_enrollment'
        - 'sealed_evaluation'
        """
        valid_roles = {"unassigned", "training", "development", "calibration_enrollment", "sealed_evaluation"}
        if new_role not in valid_roles:
            raise StethoAPIException(f"Invalid dataset_role '{new_role}'. Must be one of {valid_roles}", status_code=400)

        now = get_utc_now_iso()

        with self.db.transaction() as cur:
            cur.execute("SELECT * FROM field_recordings WHERE recording_id = ?;", (recording_id,))
            rec = cur.fetchone()
            if not rec:
                raise StethoAPIException(f"Recording '{recording_id}' not found.", status_code=404)

            current_role = rec["dataset_role"]
            current_label = rec["current_label"]

            # Role restriction 1: Calibration enrollment recordings MUST be confirmed healthy normal
            if new_role == "calibration_enrollment" and (current_label != "normal" or rec["verification_status"] != "VERIFIED"):
                raise DatasetRoleConflictException(
                    "Cannot assign role 'calibration_enrollment' to an abnormal recording or unverified recording. "
                    "Enrollment audio must represent confirmed healthy normal baseline operation."
                )

            # Role restriction 2: If recording is in a sealed manifest, prevent role change
            if current_role == "sealed_evaluation" and new_role != "sealed_evaluation":
                cur.execute("""
                SELECT manifest_id, name, recordings_json FROM dataset_manifests
                WHERE dataset_role = 'sealed_evaluation' AND is_sealed = 1;
                """)
                for manifest in cur.fetchall():
                    member_ids = json.loads(manifest["recordings_json"] or "[]")
                    if recording_id in member_ids:
                        raise ManifestSealedException(
                            f"Recording '{recording_id}' belongs to sealed evaluation manifest '{manifest['name']}' "
                            f"({manifest['manifest_id']}) and cannot be reassigned.",
                            manifest_id=manifest["manifest_id"],
                        )

            cur.execute("""
            UPDATE field_recordings
            SET dataset_role = ?, updated_at = ?
            WHERE recording_id = ?;
            """, (new_role, now, recording_id))

        return self.get_recording(recording_id)

    def create_or_seal_manifest(
        self,
        name: str,
        dataset_role: str,
        machine_category: str,
        recording_ids: List[str],
        created_by: str,
        seal_now: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and optionally seal an immutable dataset manifest with SHA-256 checksums."""
        valid_roles = {"training", "development", "calibration_enrollment", "sealed_evaluation"}
        if dataset_role not in valid_roles:
            raise StethoAPIException(f"Invalid dataset_role '{dataset_role}'. Must be one of {valid_roles}", status_code=400)

        if not recording_ids:
            raise StethoAPIException("Manifest must contain at least one recording ID.", status_code=400)

        now = get_utc_now_iso()
        manifest_id = f"mnf_{uuid.uuid4().hex[:12]}"

        with self.db.transaction() as cur:
            # 1. Verify all recordings exist and belong to dataset_role
            placeholders = ",".join("?" for _ in recording_ids)
            cur.execute(f"""
            SELECT recording_id, sha256_hash, dataset_role, current_label
            FROM field_recordings
            WHERE recording_id IN ({placeholders});
            """, recording_ids)
            records = cur.fetchall()

            found_ids = {r["recording_id"] for r in records}
            missing = set(recording_ids) - found_ids
            if missing:
                raise StethoAPIException(f"Recordings not found: {missing}", status_code=404)

            # Role boundary check
            conflicting_roles = [r["recording_id"] for r in records if r["dataset_role"] != dataset_role]
            if conflicting_roles:
                raise DatasetRoleConflictException(
                    f"Recordings {conflicting_roles} have role '{records[0]['dataset_role']}' "
                    f"which does not match manifest role '{dataset_role}'."
                )

            # If calibration enrollment, verify zero abnormal recordings
            if dataset_role == "calibration_enrollment":
                abnormal_clips = [r["recording_id"] for r in records if r["current_label"] == "abnormal"]
                if abnormal_clips:
                    raise DatasetRoleConflictException(
                        f"Calibration manifest cannot contain abnormal recordings: {abnormal_clips}"
                    )

            # 2. Compute cryptographic manifest checksum over sorted (recording_id, sha256_hash)
            sorted_entries = sorted([(r["recording_id"], r["sha256_hash"]) for r in records])
            hasher = hashlib.sha256()
            for r_id, r_hash in sorted_entries:
                hasher.update(f"{r_id}:{r_hash}\n".encode("utf-8"))
            checksum = hasher.hexdigest()

            is_sealed = 1 if seal_now else 0
            sealed_at = now if seal_now else None

            cur.execute("""
            INSERT INTO dataset_manifests (
                manifest_id, name, dataset_role, machine_category, is_sealed,
                sealed_at, sha256_checksum, recording_count, recordings_json,
                metadata_json, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                manifest_id, name, dataset_role, machine_category, is_sealed,
                sealed_at, checksum, len(recording_ids), json.dumps(recording_ids),
                json.dumps(metadata or {}), created_by, now, now
            ))

        return {
            "manifest_id": manifest_id,
            "name": name,
            "dataset_role": dataset_role,
            "machine_category": machine_category,
            "is_sealed": bool(is_sealed),
            "sealed_at": sealed_at,
            "sha256_checksum": checksum,
            "recording_count": len(recording_ids),
            "recordings": recording_ids,
            "created_by": created_by,
            "created_at": now,
        }

    def list_manifests(self, dataset_role: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all dataset manifests."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        query = "SELECT * FROM dataset_manifests"
        params = []
        if dataset_role:
            query += " WHERE dataset_role = ?"
            params.append(dataset_role)
        query += " ORDER BY created_at DESC;"
        cur.execute(query, params)
        rows = cur.fetchall()
        results = []
        for r in rows:
            m = dict(r)
            m["recordings"] = json.loads(m.get("recordings_json") or "[]")
            m["metadata"] = json.loads(m.get("metadata_json") or "{}")
            results.append(m)
        return results

    def get_manifest(self, manifest_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single manifest by ID."""
        conn = self.db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM dataset_manifests WHERE manifest_id = ?;", (manifest_id,))
        row = cur.fetchone()
        if not row:
            return None
        m = dict(row)
        m["recordings"] = json.loads(m.get("recordings_json") or "[]")
        m["metadata"] = json.loads(m.get("metadata_json") or "{}")
        return m

    def seal_manifest(self, manifest_id: str, operator_id: str) -> Dict[str, Any]:
        """Seal an existing manifest and freeze its checksum."""
        now = get_utc_now_iso()
        manifest = self.get_manifest(manifest_id)
        if not manifest:
            raise StethoAPIException(f"Manifest '{manifest_id}' not found.", status_code=404)
        if manifest["is_sealed"]:
            raise ManifestSealedException(f"Manifest '{manifest_id}' is already sealed.", manifest_id=manifest_id)

        recording_ids = manifest["recordings"]
        if not recording_ids:
            raise StethoAPIException("Cannot seal an empty manifest.", status_code=400)

        with self.db.transaction() as cur:
            placeholders = ",".join("?" for _ in recording_ids)
            cur.execute(f"SELECT recording_id, sha256_hash FROM field_recordings WHERE recording_id IN ({placeholders});", recording_ids)
            recs = cur.fetchall()

            sorted_entries = sorted([(r["recording_id"], r["sha256_hash"]) for r in recs])
            hasher = hashlib.sha256()
            for r_id, r_hash in sorted_entries:
                hasher.update(f"{r_id}:{r_hash}\n".encode("utf-8"))
            checksum = hasher.hexdigest()

            cur.execute("""
            UPDATE dataset_manifests
            SET is_sealed = 1, sealed_at = ?, sha256_checksum = ?, updated_at = ?
            WHERE manifest_id = ?;
            """, (now, checksum, now, manifest_id))

        return self.get_manifest(manifest_id)

    def verify_manifest(self, manifest_id: str) -> Dict[str, Any]:
        """Cryptographically verify all member files and overall manifest checksum against tampering."""
        manifest = self.get_manifest(manifest_id)
        if not manifest:
            raise StethoAPIException(f"Manifest '{manifest_id}' not found.", status_code=404)

        now = get_utc_now_iso()
        recording_ids = manifest["recordings"]
        if not recording_ids:
            return {
                "manifest_id": manifest_id,
                "is_valid": False,
                "tamper_detected": True,
                "reason": "Manifest has zero member recordings.",
                "verified_at": now,
            }

        conn = self.db.get_connection()
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in recording_ids)
        cur.execute(f"SELECT recording_id, sha256_hash, file_path FROM field_recordings WHERE recording_id IN ({placeholders});", recording_ids)
        rows = {r["recording_id"]: dict(r) for r in cur.fetchall()}

        missing_db = [rid for rid in recording_ids if rid not in rows]
        if missing_db:
            return {
                "manifest_id": manifest_id,
                "is_valid": False,
                "tamper_detected": True,
                "reason": f"Recordings missing from database: {missing_db}",
                "verified_at": now,
            }

        tampered_files = []
        for rid, rec in rows.items():
            f_path = Path(rec["file_path"])
            if not f_path.exists():
                tampered_files.append(f"{rid} (missing on disk)")
                continue
            with open(f_path, "rb") as f_in:
                disk_hash = hashlib.sha256(f_in.read()).hexdigest()
            if disk_hash != rec["sha256_hash"]:
                tampered_files.append(f"{rid} (hash mismatch: expected {rec['sha256_hash'][:8]}, disk has {disk_hash[:8]})")

        if tampered_files:
            return {
                "manifest_id": manifest_id,
                "is_valid": False,
                "tamper_detected": True,
                "reason": f"File tamper detected: {tampered_files}",
                "verified_at": now,
            }

        sorted_entries = sorted([(r["recording_id"], r["sha256_hash"]) for r in rows.values()])
        hasher = hashlib.sha256()
        for r_id, r_hash in sorted_entries:
            hasher.update(f"{r_id}:{r_hash}\n".encode("utf-8"))
        recomputed = hasher.hexdigest()

        checksum_matches = (recomputed == manifest["sha256_checksum"])
        return {
            "manifest_id": manifest_id,
            "name": manifest["name"],
            "dataset_role": manifest["dataset_role"],
            "is_sealed": bool(manifest["is_sealed"]),
            "is_valid": checksum_matches,
            "tamper_detected": not checksum_matches,
            "stored_checksum": manifest["sha256_checksum"],
            "recomputed_checksum": recomputed,
            "total_recordings": len(recording_ids),
            "verified_at": now,
        }

    def get_recording_audio_path(self, recording_id: str) -> Path:
        """Fetch and validate audio file path for secure audio delivery."""
        rec = self.get_recording(recording_id)
        if not rec:
            raise StethoAPIException(f"Recording '{recording_id}' not found.", status_code=404)
        file_path = Path(rec["file_path"]).resolve()
        base_dir = self.storage_base_dir.resolve()
        if not str(file_path).startswith(str(base_dir)):
            raise StethoAPIException("Access denied: File outside storage root.", status_code=403)
        if not file_path.exists():
            raise StethoAPIException(f"Audio file for recording '{recording_id}' not found on disk.", status_code=404)
        return file_path

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Aggregate live field data metrics and evaluate independent pilot readiness."""
        conn = self.db.get_connection()
        cur = conn.cursor()

        # 1. Total imported & by category
        cur.execute("SELECT COUNT(*) FROM field_recordings;")
        total_imported = cur.fetchone()[0]

        cur.execute("SELECT machine_category, COUNT(*) FROM field_recordings GROUP BY machine_category;")
        by_category = {row[0]: row[1] for row in cur.fetchall()}

        # 2. By machine ID
        cur.execute("SELECT machine_id, COUNT(*) FROM field_recordings GROUP BY machine_id;")
        by_machine_id = {row[0]: row[1] for row in cur.fetchall()}

        # 3. By Ground Truth Label
        cur.execute("SELECT current_label, COUNT(*) FROM field_recordings GROUP BY current_label;")
        by_label = {row[0]: row[1] for row in cur.fetchall()}

        verified_normal = by_label.get("normal", 0)
        verified_abnormal = by_label.get("abnormal", 0)
        uncertain_count = by_label.get("uncertain", 0)
        unlabelled_count = by_label.get("unlabelled", 0)

        # 4. By Verification Status
        cur.execute("SELECT verification_status, COUNT(*) FROM field_recordings GROUP BY verification_status;")
        by_verification_status = {row[0]: row[1] for row in cur.fetchall()}

        # 5. By Dataset Role
        cur.execute("SELECT dataset_role, COUNT(*) FROM field_recordings GROUP BY dataset_role;")
        by_dataset_role = {row[0]: row[1] for row in cur.fetchall()}

        # 6. By Sensor & Channel
        cur.execute("SELECT sensor_id, channel_index, COUNT(*) FROM field_recordings GROUP BY sensor_id, channel_index;")
        by_sensor_channel = {f"{row[0]}_ch{row[1]}": row[2] for row in cur.fetchall()}

        # 7. Auditable Import Rejection History
        cur.execute("SELECT status, COUNT(*) FROM field_import_history GROUP BY status;")
        import_stats = {row[0]: row[1] for row in cur.fetchall()}
        duplicate_count = import_stats.get("DUPLICATE_REJECTED", 0)
        rejection_count = import_stats.get("VALIDATION_FAILED", 0)

        # 8. Missing Metadata Count
        cur.execute("""
        SELECT COUNT(*) FROM field_recordings
        WHERE operating_condition IS NULL OR operating_condition = ''
           OR load_level IS NULL OR load_level = ''
           OR sensor_id IS NULL OR sensor_id = '';
        """)
        missing_metadata_count = cur.fetchone()[0]

        # 9. Independent Evaluation Readiness Assessment
        # Evaluation readiness requires at least 10 verified normal and 10 verified abnormal
        # recordings strictly sealed in the 'sealed_evaluation' dataset role.
        cur.execute("""
        SELECT current_label, COUNT(*) FROM field_recordings
        WHERE dataset_role = 'sealed_evaluation'
        GROUP BY current_label;
        """)
        sealed_counts = {row[0]: row[1] for row in cur.fetchall()}
        sealed_normal = sealed_counts.get("normal", 0)
        sealed_abnormal = sealed_counts.get("abnormal", 0)

        is_independent_evaluation_ready = (sealed_normal >= 10 and sealed_abnormal >= 10)

        readiness_status = {
            "is_ready": is_independent_evaluation_ready,
            "sealed_normal_recordings": sealed_normal,
            "sealed_abnormal_recordings": sealed_abnormal,
            "minimum_required_per_class": 10,
            "verdict": (
                "READY_FOR_SEALED_EVALUATION"
                if is_independent_evaluation_ready
                else "INSUFFICIENT_INDEPENDENT_EVIDENCE"
            ),
            "warning": (
                None
                if is_independent_evaluation_ready
                else "INSUFFICIENT INDEPENDENT EVIDENCE: Genuinely independent field recordings are currently "
                     f"insufficient for a statistically sound pilot evaluation (found {sealed_normal} normal, "
                     f"{sealed_abnormal} abnormal in sealed evaluation; minimum required is 10 per class). "
                     "Field data collection in progress."
            ),
        }

        return {
            "total_imported": total_imported,
            "by_category": by_category,
            "by_machine_id": by_machine_id,
            "verified_normal": verified_normal,
            "verified_abnormal": verified_abnormal,
            "uncertain_count": uncertain_count,
            "unlabelled_count": unlabelled_count,
            "by_verification_status": by_verification_status,
            "by_dataset_role": by_dataset_role,
            "by_sensor_channel": by_sensor_channel,
            "duplicates_detected": duplicate_count,
            "rejections_count": rejection_count,
            "missing_metadata_count": missing_metadata_count,
            "independent_evaluation_readiness": readiness_status,
        }

    def get_collection_protocol_checklist(self) -> Dict[str, Any]:
        """Return the standardized field operator checklist and guidelines."""
        return {
            "title": "Stetho Field Acoustic Data Collection Checklist",
            "version": "1.0.0",
            "specifications": {
                "format": "WAV (Linear PCM, 16-bit or 24-bit)",
                "sample_rate": "16,000 Hz minimum (48,000 Hz supported via polyphase resampler)",
                "duration": "10.0 seconds per recording (continuous non-stop)",
                "channels": "1 channel (primary capsule) or multi-channel array",
            },
            "steps": [
                {
                    "step": 1,
                    "title": "Safety & Site Inspection",
                    "action": "Ensure personal protective equipment (PPE) is worn. Confirm mechanical equipment is safely accessible without contact with rotating components.",
                },
                {
                    "step": 2,
                    "title": "Transducer Mounting & Geometry",
                    "action": "Rigidly clamp or magnetically mount microphone at 10 cm +/- 2 cm from primary bearing housing. Record sensor serial number and orientation (0 deg line of sight). Take context photo.",
                },
                {
                    "step": 3,
                    "title": "Machine Operating State Verification",
                    "action": "Record current RPM, mechanical load percentage (kW or Amps), and operational state (nominal, idle, peak load).",
                },
                {
                    "step": 4,
                    "title": "Healthy Baseline Enrollment (N=10)",
                    "action": "Capture exactly 10 non-overlapping clips of 10.0 seconds each during verified healthy operation. Space over 2 hours if feasible.",
                },
                {
                    "step": 5,
                    "title": "Defect Capture & Ground-Truth Triangulation",
                    "action": "For anomalous machines, cross-verify defect with ISO 10816-3 vibration order analysis (overall velocity mm/s RMS) and infrared thermal imaging. Attach report ID.",
                },
                {
                    "step": 6,
                    "title": "File Integrity Verification",
                    "action": "Verify files play cleanly without static or dead silence. Ingest via Stetho Field Ingest API to calculate and seal cryptographic SHA-256 hashes.",
                },
            ],
            "evidence_sources": [
                "VIBRATION_ORDER_ANALYSIS",
                "THERMAL_SCAN",
                "OIL_ANALYSIS",
                "TEARDOWN_INSPECTION",
                "VISUAL_INSPECTION",
                "OPERATIONAL_LOG",
            ],
        }
