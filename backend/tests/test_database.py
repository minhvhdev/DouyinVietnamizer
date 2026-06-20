from pathlib import Path

from dv_backend.database import Database


def test_migration_creates_required_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    database.migrate()

    rows = database.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()

    assert {"jobs", "job_steps", "settings", "events", "runtime_reports"} <= {row["name"] for row in rows}
    job_columns = {
        row["name"]
        for row in database.connection.execute("PRAGMA table_info(jobs)").fetchall()
    }
    assert "title_vi" in job_columns

