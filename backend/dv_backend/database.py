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
    name TEXT NOT NULL UNIQUE,
    wav_filename TEXT NOT NULL,
    transcript TEXT,
    created_at TEXT NOT NULL
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
        self.connection.commit()
