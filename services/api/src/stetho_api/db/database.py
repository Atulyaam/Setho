"""Database persistence layer and schema management for Stetho API.

Provides ACID persistence for users, machine profiles, enrollments, analysis records,
and asynchronous job tracking using a thread-safe SQLite backend with automated
migrations and schema initialization.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger("stetho.database")


def get_utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """Thread-safe SQLite database manager with schema initialization and connection pooling."""

    def __init__(self, db_path: Union[str, Path] = "./data/storage/stetho.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode; transactions managed explicitly
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for fast concurrent reads and writes
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager providing an explicit atomic transaction."""
        conn = self.get_connection()
        with self._lock:
            conn.execute("BEGIN IMMEDIATE;")
            cursor = conn.cursor()
            try:
                yield cursor
                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error(f"Transaction rolled back due to error: {e}")
                raise e

    def initialize_schema(self) -> None:
        """Create tables and indices if they do not exist, and apply migrations."""
        from stetho_api.auth.password import hash_password

        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()

            # 1. Users Table (Authentication & Access Control)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'inspector',
                created_at TEXT NOT NULL
            );
            """)

            # 2. Machines Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                machine_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT DEFAULT '',
                operating_condition TEXT DEFAULT '',
                selected_model_policy TEXT NOT NULL DEFAULT 'stetho_msae',
                enrollment_status TEXT NOT NULL DEFAULT 'uncalibrated',
                enrolled_recording_count INTEGER NOT NULL DEFAULT 0,
                calibration_status TEXT NOT NULL DEFAULT 'uncalibrated',
                calibration_profile TEXT DEFAULT NULL,
                calibration_version TEXT NOT NULL DEFAULT '1.0.0',
                owner_id TEXT DEFAULT 'usr_admin',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users(user_id)
            );
            """)

            # Automated Migrations for existing machines table
            existing_cols = [r[1] for r in cursor.execute("PRAGMA table_info(machines);").fetchall()]
            if "owner_id" not in existing_cols:
                cursor.execute("ALTER TABLE machines ADD COLUMN owner_id TEXT DEFAULT 'usr_admin';")
            if "calibration_profile" not in existing_cols:
                cursor.execute("ALTER TABLE machines ADD COLUMN calibration_profile TEXT DEFAULT NULL;")
            if "calibration_version" not in existing_cols:
                cursor.execute("ALTER TABLE machines ADD COLUMN calibration_version TEXT NOT NULL DEFAULT '1.0.0';")

            # 3. Enrollments Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                enrollment_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                upload_id TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                sample_rate INTEGER NOT NULL,
                channels INTEGER NOT NULL,
                quality_status TEXT NOT NULL,
                is_confirmed_normal INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
            );
            """)

            existing_enr_cols = [r[1] for r in cursor.execute("PRAGMA table_info(enrollments);").fetchall()]
            if "content_hash" not in existing_enr_cols:
                cursor.execute("ALTER TABLE enrollments ADD COLUMN content_hash TEXT DEFAULT NULL;")

            # 4. Analyses Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                analysis_status TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                decision TEXT NOT NULL,
                anomaly_score REAL,
                operating_threshold REAL,
                theta_lower REAL,
                theta_upper REAL,
                is_provisional INTEGER NOT NULL DEFAULT 1,
                processing_duration_ms REAL,
                reasons TEXT,
                metrics TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
            );
            """)

            # 5. Asynchronous Analysis Jobs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                job_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_pct INTEGER NOT NULL DEFAULT 0,
                analysis_id TEXT,
                error_message TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
            );
            """)

            # 6. Machine Acoustic Fingerprints Table (Sprint 6A Feature 1)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS machine_fingerprints (
                fingerprint_id TEXT PRIMARY KEY,
                machine_id TEXT UNIQUE NOT NULL,
                fingerprint_version TEXT NOT NULL DEFAULT '1.0.0',
                spectral_centroid_hz REAL,
                sub_bass_energy_db REAL,
                low_energy_db REAL,
                mid_energy_db REAL,
                high_energy_db REAL,
                baseline_reconstruction_error REAL,
                baseline_quality_score REAL,
                sample_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
            );
            """)

            # 7. Machine QR Identity Tokens Table (Sprint 6A Feature 6)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS machine_qr_tokens (
                token_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                qr_token TEXT UNIQUE NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id) ON DELETE CASCADE
            );
            """)

            # 8. Assets Table (Sprint 7 Asset Management)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT DEFAULT '',
                operating_condition TEXT DEFAULT '',
                assigned_model TEXT NOT NULL DEFAULT 'cascade_ensemble',
                assigned_model_version TEXT NOT NULL DEFAULT '1.0.0',
                calibration_status TEXT NOT NULL DEFAULT 'uncalibrated',
                calibration_version TEXT NOT NULL DEFAULT '1.0.0',
                enrolled_recording_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                owner_id TEXT DEFAULT 'usr_admin',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (owner_id) REFERENCES users(user_id)
            );
            """)

            # 9. Calibration Profiles Table (Sprint 7 Few-Shot Calibration)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_profiles (
                profile_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                calibration_version TEXT NOT NULL,
                operating_threshold REAL NOT NULL,
                theta_lower REAL NOT NULL,
                theta_upper REAL NOT NULL,
                baseline_mean REAL NOT NULL,
                baseline_std REAL NOT NULL,
                enrolled_count INTEGER NOT NULL,
                latent_mean_json TEXT DEFAULT NULL,
                latent_cov_json TEXT DEFAULT NULL,
                scores_json TEXT DEFAULT '[]',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
            );
            """)

            # 10. Calibration History Table (Sprint 7 Audit & Rollback)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_history (
                history_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                previous_version TEXT,
                new_version TEXT NOT NULL,
                previous_threshold REAL,
                new_threshold REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                operator_id TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
            );
            """)

            # 11. Inference Results Table (Sprint 7 Standardized Inference)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_results (
                result_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                anomaly_score REAL,
                operating_threshold REAL,
                anomaly_decision TEXT NOT NULL,
                processing_latency_ms REAL NOT NULL,
                calibration_status TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                is_provisional INTEGER NOT NULL DEFAULT 1,
                reasons_json TEXT DEFAULT '[]',
                metrics_json TEXT DEFAULT '{}',
                audio_storage_path TEXT,
                audio_hash TEXT,
                operator_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
            );
            """)

            # 12. Model Registry Records Table (Sprint 7 Versioned Registry)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_registry_records (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                architecture TEXT NOT NULL,
                version TEXT NOT NULL,
                category TEXT NOT NULL,
                profile TEXT NOT NULL,
                runtime_engine TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                sha256_hash TEXT NOT NULL,
                parameter_count INTEGER,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """)

            # 13. System Events Table (Sprint 7 Audit Logging)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                component TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'INFO',
                message TEXT NOT NULL,
                details_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            """)

            # Automated Migrations for Sprint 6A
            mach_cols = [r[1] for r in cursor.execute("PRAGMA table_info(machines);").fetchall()]
            if "last_inspection_at" not in mach_cols:
                cursor.execute("ALTER TABLE machines ADD COLUMN last_inspection_at TEXT DEFAULT NULL;")
            if "priority_score" not in mach_cols:
                cursor.execute("ALTER TABLE machines ADD COLUMN priority_score REAL DEFAULT 0.0;")
            if "priority_tier" not in mach_cols:
                cursor.execute("ALTER TABLE machines ADD COLUMN priority_tier TEXT DEFAULT 'NORMAL';")

            ana_cols = [r[1] for r in cursor.execute("PRAGMA table_info(analyses);").fetchall()]
            if "spectral_drift_pct" not in ana_cols:
                cursor.execute("ALTER TABLE analyses ADD COLUMN spectral_drift_pct REAL DEFAULT 0.0;")
            if "dominant_band_deviation" not in ana_cols:
                cursor.execute("ALTER TABLE analyses ADD COLUMN dominant_band_deviation TEXT DEFAULT NULL;")
            if "maintenance_urgency" not in ana_cols:
                cursor.execute("ALTER TABLE analyses ADD COLUMN maintenance_urgency TEXT DEFAULT 'NORMAL';")

            # Indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_machines_owner ON machines(owner_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_machine ON enrollments(machine_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_machine ON analyses(machine_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fingerprints_machine ON machine_fingerprints(machine_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_qr_tokens ON machine_qr_tokens(qr_token);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_owner ON assets(owner_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_category ON assets(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cal_profiles_asset ON calibration_profiles(asset_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cal_history_asset ON calibration_history(asset_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inf_results_asset ON inference_results(asset_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sys_events_comp ON system_events(component);")

            # Seed default admin and inspector accounts if users table is empty
            cursor.execute("SELECT COUNT(*) FROM users;")
            user_count = cursor.fetchone()[0]
            now = get_utc_now_iso()
            if user_count == 0:
                is_prod = os.environ.get("ENVIRONMENT", "development").lower() == "production"
                admin_pw = os.environ.get("STETHO_ADMIN_PASSWORD")
                inspector_pw = os.environ.get("STETHO_INSPECTOR_PASSWORD")

                if is_prod and (not admin_pw or not inspector_pw):
                    logger.warning(
                        "Production environment detected: Default seed accounts omitted. "
                        "Configure STETHO_ADMIN_PASSWORD and STETHO_INSPECTOR_PASSWORD or use /api/v1/auth/register."
                    )
                else:
                    admin_pw = admin_pw or "StethoAdmin2026!"
                    inspector_pw = inspector_pw or "StethoInspector2026!"
                    admin_hash = hash_password(admin_pw)
                    inspector_hash = hash_password(inspector_pw)
                    default_users = [
                        ("usr_admin", "admin", admin_hash, "admin", now),
                        ("usr_inspector", "inspector", inspector_hash, "inspector", now),
                    ]
                    cursor.executemany("""
                    INSERT INTO users (user_id, username, hashed_password, role, created_at)
                    VALUES (?, ?, ?, ?, ?);
                    """, default_users)
                    conn.commit()
                    logger.info("Configured default 'admin' and 'inspector' accounts in database")

            # Seed pre-calibrated MIMII benchmark machines if table is empty
            cursor.execute("SELECT COUNT(*) FROM machines;")
            count = cursor.fetchone()[0]
            if count == 0:
                default_machines = [
                    ("id_00", "Industrial Electric Fan ID-00", "fan", "MIMII 1.0 Benchmark Fan Unit 0", "Normal speed, 0dB ambient noise", "stetho_msae", "provisional", 10, "provisional", None, "1.0.0", "usr_admin", now, now),
                    ("id_02", "Industrial Electric Fan ID-02", "fan", "MIMII 1.0 Benchmark Fan Unit 2", "Normal speed, 0dB ambient noise", "stetho_msae", "provisional", 10, "provisional", None, "1.0.0", "usr_admin", now, now),
                    ("id_04", "Industrial Electric Fan ID-04", "fan", "MIMII 1.0 Benchmark Fan Unit 4", "Normal speed, 0dB ambient noise", "stetho_msae", "provisional", 10, "provisional", None, "1.0.0", "usr_admin", now, now),
                    ("id_06", "Industrial Electric Fan ID-06", "fan", "MIMII 1.0 Benchmark Fan Unit 6", "Normal speed, 0dB ambient noise", "stetho_msae", "provisional", 10, "provisional", None, "1.0.0", "usr_admin", now, now),
                ]
                cursor.executemany("""
                INSERT INTO machines (
                    machine_id, name, category, description, operating_condition,
                    selected_model_policy, enrollment_status, enrolled_recording_count,
                    calibration_status, calibration_profile, calibration_version, owner_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, default_machines)
                conn.commit()
                logger.info(f"Seeded {len(default_machines)} benchmark machine profiles in database")

            # Seed assets from machines table if assets table is empty
            cursor.execute("SELECT COUNT(*) FROM assets;")
            asset_count = cursor.fetchone()[0]
            if asset_count == 0:
                cursor.execute("""
                INSERT OR IGNORE INTO assets (
                    asset_id, name, category, description, operating_condition,
                    assigned_model, assigned_model_version, calibration_status,
                    calibration_version, enrolled_recording_count, metadata_json,
                    owner_id, created_at, updated_at
                ) SELECT
                    machine_id, name, category, description, operating_condition,
                    selected_model_policy, '1.0.0', calibration_status,
                    calibration_version, enrolled_recording_count, '{}',
                    owner_id, created_at, updated_at
                FROM machines;
                """)
                conn.commit()
                logger.info("Synchronized benchmark machines to assets table")

            # Seed default production models in model_registry_records if empty
            cursor.execute("SELECT COUNT(*) FROM model_registry_records;")
            if cursor.fetchone()[0] == 0:
                default_models = [
                    (
                        "mod_compact_student_int8_v1",
                        "Distilled Compact Student-AE (INT8)",
                        "CompactStudentAE",
                        "1.0.0",
                        "universal",
                        "edge_real_time",
                        "onnx_int8",
                        "./ml/experiments/research_artifacts/phase8/exported_models/student_ae_int8.onnx",
                        "d4c0bbf2f07ab2374e2a3928198f39942a781f8f3c7e7bdfc8369ec23748be71",
                        27000,
                        1,
                        now,
                    ),
                    (
                        "mod_resnet_ae_v1",
                        "Optimized ResNet-AE (FP32)",
                        "ResidualCNNAE",
                        "1.0.0",
                        "fan",
                        "cloud_high_accuracy",
                        "pytorch",
                        "./ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt",
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        742000,
                        1,
                        now,
                    ),
                    (
                        "mod_multitask_resnet_v1",
                        "MultiTask ResNet-AE (Zero-Shot)",
                        "MultiTaskResNetAE",
                        "1.0.0",
                        "fan",
                        "zero_shot",
                        "pytorch",
                        "./ml/models/research/multitask_resnet_ae.py",
                        "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
                        810000,
                        1,
                        now,
                    ),
                    (
                        "mod_cascade_ensemble_v1",
                        "Margin-Gated Cascade Ensemble",
                        "MarginGatedCascade",
                        "1.0.0",
                        "fan",
                        "hybrid_cascade",
                        "hybrid",
                        "./ml/models/research/margin_cascade.py",
                        "b2c3d4e5f6a10718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
                        769000,
                        1,
                        now,
                    ),
                    (
                        "mod_stetho_msae_v1",
                        "Stetho Multi-Scale Autoencoder",
                        "StethoMSAE",
                        "1.0.0",
                        "fan",
                        "legacy_production",
                        "pytorch",
                        "./ml/artifacts/stetho_msae/id_00/model.pt",
                        "c3d4e5f6a1b20718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
                        559873,
                        1,
                        now,
                    ),
                ]
                cursor.executemany("""
                INSERT INTO model_registry_records (
                    model_id, model_name, architecture, version, category,
                    profile, runtime_engine, checkpoint_path, sha256_hash,
                    parameter_count, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, default_models)
                conn.commit()
                logger.info(f"Seeded {len(default_models)} production model records in model_registry_records")

            conn.close()


# Application singleton
_db_instance: Optional[DatabaseManager] = None
_db_lock = threading.Lock()


def get_db(db_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """Return database manager singleton."""
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            if db_path is None:
                from stetho_api.config import get_settings
                settings = get_settings()
                db_file = settings.STORAGE_LOCAL_DIR / "stetho.db"
            else:
                db_file = Path(db_path)
            _db_instance = DatabaseManager(db_file)
        return _db_instance
