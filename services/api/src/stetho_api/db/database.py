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

            # Indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_machines_owner ON machines(owner_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_machine ON enrollments(machine_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_machine ON analyses(machine_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);")

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
