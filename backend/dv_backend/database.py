from pathlib import Path
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    title TEXT,
    title_vi TEXT,
    status TEXT NOT NULL,
    current_step TEXT,
    last_error_code TEXT,
    last_error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_steps (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    status TEXT NOT NULL,
    checkpoint_path TEXT,
    error_code TEXT,
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER,
    PRIMARY KEY (job_id, name)
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    job_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cloned_voices (
    id TEXT PRIMARY KEY,
    backend TEXT NOT NULL DEFAULT 'omnivoice',
    name TEXT NOT NULL,
    wav_filename TEXT NOT NULL,
    transcript TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(backend, name)
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA foreign_keys = ON")

    def migrate(self) -> None:
        self.connection.executescript(SCHEMA)
        job_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "title_vi" not in job_columns:
            self.connection.execute("ALTER TABLE jobs ADD COLUMN title_vi TEXT")
        step_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(job_steps)").fetchall()
        }
        if "duration_ms" not in step_columns:
            self.connection.execute("ALTER TABLE job_steps ADD COLUMN duration_ms INTEGER")
        voice_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(cloned_voices)").fetchall()
        }
        if "transcript" not in voice_columns:
            self.connection.execute("ALTER TABLE cloned_voices ADD COLUMN transcript TEXT")
        self._migrate_cloned_voices_backend_schema()
        self._migrate_cloned_voices_profile_schema()
        self._migrate_voice_calibration_jobs_schema()
        self.connection.commit()

    def _migrate_cloned_voices_profile_schema(self) -> None:
        voice_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(cloned_voices)").fetchall()
        }
        additions = {
            "voice_status": "TEXT NOT NULL DEFAULT 'ready'",
            "duration_profile_status": "TEXT NOT NULL DEFAULT 'not_started'",
            "duration_profile_key": "TEXT",
            "duration_profile_quality": "TEXT",
            "duration_profile_sample_count": "INTEGER NOT NULL DEFAULT 0",
            "last_calibrated_at": "TEXT",
            "active_calibration_job_id": "TEXT",
        }
        for column, ddl in additions.items():
            if column not in voice_columns:
                self.connection.execute(f"ALTER TABLE cloned_voices ADD COLUMN {column} {ddl}")

    def _migrate_voice_calibration_jobs_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS voice_calibration_jobs (
                job_id TEXT PRIMARY KEY,
                voice_id TEXT NOT NULL,
                voice_identity_key TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                sample_total INTEGER NOT NULL DEFAULT 0,
                sample_completed INTEGER NOT NULL DEFAULT 0,
                sample_accepted INTEGER NOT NULL DEFAULT 0,
                sample_rejected INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    def _migrate_cloned_voices_backend_schema(self) -> None:
        voice_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(cloned_voices)").fetchall()
        }
        if "backend" in voice_columns:
            return
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cloned_voices_new (
                id TEXT PRIMARY KEY,
                backend TEXT NOT NULL DEFAULT 'omnivoice',
                name TEXT NOT NULL,
                wav_filename TEXT NOT NULL,
                transcript TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(backend, name)
            );
            INSERT INTO cloned_voices_new (id, backend, name, wav_filename, transcript, created_at)
            SELECT id, 'omnivoice', name, wav_filename, transcript, created_at
            FROM cloned_voices;
            DROP TABLE cloned_voices;
            ALTER TABLE cloned_voices_new RENAME TO cloned_voices;
            """
        )
