import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "aero_guard.db"


def get_connection():
    """Open a connection to the local SQLite database."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    """Create application tables and preserve existing demo records."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with closing(get_connection()) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                subsystem TEXT NOT NULL,
                health INTEGER NOT NULL
                    CHECK (health BETWEEN 0 AND 100),
                status TEXT NOT NULL,
                tone TEXT NOT NULL,
                operating_window TEXT NOT NULL,
                is_demo INTEGER NOT NULL DEFAULT 1
                    CHECK (is_demo IN (0, 1)),
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS work_orders (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                task TEXT NOT NULL,
                priority TEXT NOT NULL
                    CHECK (priority IN ('Low', 'Medium', 'High')),
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            );

            CREATE TABLE IF NOT EXISTS spare_parts (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                stock INTEGER NOT NULL CHECK (stock >= 0),
                required INTEGER NOT NULL CHECK (required >= 0),
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evidence_audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                engine_unit INTEGER NOT NULL,
                last_observed_cycle INTEGER NOT NULL,
                predicted_rul_cycles REAL NOT NULL,
                model_disagreement_cycles REAL NOT NULL,
                evidence_status TEXT NOT NULL,
                decision TEXT NOT NULL,
                evidence_completeness TEXT NOT NULL,
                review_reasons_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                source_dataset TEXT NOT NULL,
                ensemble_size INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS engine_asset_mappings (
                engine_unit INTEGER PRIMARY KEY CHECK (engine_unit BETWEEN 1 AND 100),
                asset_id TEXT NOT NULL,
                mapping_type TEXT NOT NULL CHECK (mapping_type IN ('DEMO_ONLY', 'DOCUMENTED')),
                mapping_note TEXT NOT NULL,
                mapping_reference TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            );

            CREATE INDEX IF NOT EXISTS idx_evidence_audit_run
                ON evidence_audit_events(run_id, engine_unit);

            CREATE INDEX IF NOT EXISTS idx_evidence_audit_created
                ON evidence_audit_events(created_at);

            CREATE INDEX IF NOT EXISTS idx_evidence_audit_engine
                ON evidence_audit_events(engine_unit, id);
            """
        )

        # Insert sample assets only when the table is empty.
        asset_count = connection.execute(
            "SELECT COUNT(*) FROM assets"
        ).fetchone()[0]

        if asset_count == 0:
            connection.executemany(
                """
                INSERT INTO assets
                (id, subsystem, health, status, tone,
                 operating_window, is_demo, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                [
                    (
                        "AG-014", "Propulsion system", 92,
                        "Healthy", "green", "120 cycles", now
                    ),
                    (
                        "AG-021", "Hydraulic system", 67,
                        "Review", "amber", "48 cycles", now
                    ),
                    (
                        "AG-008", "Power system", 96,
                        "Healthy", "green", "156 cycles", now
                    ),
                    (
                        "AG-032", "Sensor integrity", 74,
                        "Monitoring", "blue", "72 cycles", now
                    ),
                ],
            )

        # Insert example work orders only once.
        order_count = connection.execute(
            "SELECT COUNT(*) FROM work_orders"
        ).fetchone()[0]

        if order_count == 0:
            connection.executemany(
                """
                INSERT INTO work_orders
                (id, asset_id, task, priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "WO-1042", "AG-021",
                        "Hydraulic subsystem inspection",
                        "High", "Awaiting review", now
                    ),
                    (
                        "WO-1043", "AG-032",
                        "Sensor integrity verification",
                        "Medium", "Scheduled", now
                    ),
                    (
                        "WO-1044", "AG-014",
                        "Routine condition assessment",
                        "Low", "Planned", now
                    ),
                ],
            )

        # Insert sample spare inventory only once.
        stock_count = connection.execute(
            "SELECT COUNT(*) FROM spare_parts"
        ).fetchone()[0]

        if stock_count == 0:
            connection.executemany(
                """
                INSERT INTO spare_parts
                (code, name, stock, required, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("SP-204", "Hydraulic filter", 8, 2, now),
                    ("SP-118", "Pressure sensor", 3, 2, now),
                    ("SP-306", "Inspection seal kit", 0, 1, now),
                ],
            )

        connection.commit()


def record_evidence_audit_events(events):
    """
    Append a complete evidence-gate run to the audit table.

    Each event must contain the fields used by the INSERT below.
    Existing audit records are never overwritten.
    """
    if not events:
        return

    with closing(get_connection()) as connection:
        connection.executemany(
            """
            INSERT INTO evidence_audit_events (
                run_id,
                evaluated_at,
                engine_unit,
                last_observed_cycle,
                predicted_rul_cycles,
                model_disagreement_cycles,
                evidence_status,
                decision,
                evidence_completeness,
                review_reasons_json,
                evidence_json,
                source_dataset,
                ensemble_size,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event["run_id"],
                    event["evaluated_at"],
                    event["engine_unit"],
                    event["last_observed_cycle"],
                    event["predicted_rul_cycles"],
                    event["model_disagreement_cycles"],
                    event["evidence_status"],
                    event["decision"],
                    event["evidence_completeness"],
                    event["review_reasons_json"],
                    event["evidence_json"],
                    event["source_dataset"],
                    event["ensemble_size"],
                    event["created_at"],
                )
                for event in events
            ],
        )

        connection.commit()


if __name__ == "__main__":
    init_db()
    print("AERO-GUARD SQLite database initialized.")
    print(f"Database location: {DB_PATH}")