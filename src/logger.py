"""
logger.py
=========
Every time the LSTM makes a prediction, this logs it to a SQLite database.
Tiny file, but it's the foundation everything else is built on.

USAGE (called automatically by realtime_inference.py):
    from src.logger import ActivityLogger
    logger = ActivityLogger()
    logger.log("pill_taking", confidence=0.92, person_id="mrs_tan")
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/carewatch.db"


class ActivityLogger:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create table if it doesn't exist yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   TEXT    NOT NULL,
                    timestamp   TEXT    NOT NULL,
                    date        TEXT    NOT NULL,
                    hour        INTEGER NOT NULL,
                    minute      INTEGER NOT NULL,
                    activity    TEXT    NOT NULL,
                    confidence  REAL    NOT NULL
                )
            """)
            conn.commit()

    def log(self, activity: str, confidence: float, person_id: str = "resident"):
        """Log one activity event."""
        now = datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO activity_log
                    (person_id, timestamp, date, hour, minute, activity, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                person_id,
                now.isoformat(),
                now.strftime("%Y-%m-%d"),
                now.hour,
                now.minute,
                activity,
                round(confidence, 4),
            ))
            conn.commit()

    def get_today(self, person_id: str = "resident") -> list[dict]:
        """Return all logs for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM activity_log
                WHERE person_id = ? AND date = ?
                ORDER BY timestamp ASC
            """, (person_id, today)).fetchall()
        return [dict(r) for r in rows]

    def get_last_n_days(self, n: int = 7, person_id: str = "resident") -> list[dict]:
        """Return all logs for the last n days."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM activity_log
                WHERE person_id = ?
                ORDER BY timestamp DESC
                LIMIT 10000
            """, (person_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_last_activity(self, person_id: str = "resident") -> dict | None:
        """Return the single most recent log entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM activity_log
                WHERE person_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (person_id,)).fetchone()
        return dict(row) if row else None