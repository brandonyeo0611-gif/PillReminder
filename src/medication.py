import sqlite3
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional

DB_PATH = "data/carewatch.db"


def _parse_hhmm(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(hour=int(h), minute=int(m))


class MedicationRepo:
    """
    Lightweight repository around SQLite for:
      - medication schedules (planned doses)
      - medication events (detected / manual intake or missed doses)
      - a small medication-specific risk component

    This is intentionally simple and hackathon-friendly.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── Schema ──────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS medication_schedule (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id     TEXT    NOT NULL,
                    medication_name TEXT  NOT NULL,
                    dose          TEXT,
                    time_of_day   TEXT    NOT NULL,   -- "HH:MM"
                    tolerance_min INTEGER NOT NULL DEFAULT 30,
                    illness_hint  TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS medication_event (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id     TEXT    NOT NULL,
                    medication_name TEXT  NOT NULL,
                    timestamp     TEXT    NOT NULL,   -- ISO
                    scheduled_id  INTEGER,
                    on_time       INTEGER NOT NULL,   -- 1 = on time, 0 = late/missed
                    source        TEXT    NOT NULL    -- "ai", "manual", "missed"
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS medication_risk (
                    person_id  TEXT PRIMARY KEY,
                    risk_score INTEGER NOT NULL,
                    updated_at TEXT    NOT NULL
                )
                """
            )
            conn.commit()

    # ── Schedules CRUD ─────────────────────────────────────────────────────
    def list_schedules(self, person_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM medication_schedule
                WHERE person_id = ?
                ORDER BY time_of_day ASC, id ASC
                """,
                (person_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_schedule(self, person_id: str, payload) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO medication_schedule
                    (person_id, medication_name, dose, time_of_day, tolerance_min, illness_hint)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    payload.medication_name,
                    payload.dose,
                    payload.time_of_day,
                    payload.tolerance_min,
                    payload.illness_hint,
                ),
            )
            schedule_id = cur.lastrowid
            conn.commit()

            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM medication_schedule WHERE id = ?", (schedule_id,)
            ).fetchone()
        return dict(row) if row else {}

    def delete_schedule(self, person_id: str, schedule_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM medication_schedule WHERE person_id = ? AND id = ?",
                (person_id, schedule_id),
            )
            conn.commit()

    # ── Events + Risk ──────────────────────────────────────────────────────
    def _get_or_init_risk(self, person_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT risk_score FROM medication_risk WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            if row:
                return int(row["risk_score"])

            # Initialise with neutral risk 0
            now_iso = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO medication_risk (person_id, risk_score, updated_at) VALUES (?, ?, ?)",
                (person_id, 0, now_iso),
            )
            conn.commit()
            return 0

    def _set_risk(self, person_id: str, new_score: int) -> int:
        new_score = max(0, min(100, int(new_score)))
        now_iso = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO medication_risk (person_id, risk_score, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    risk_score = excluded.risk_score,
                    updated_at = excluded.updated_at
                """,
                (person_id, new_score, now_iso),
            )
            conn.commit()
        return new_score

    def get_medication_risk(self, person_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT risk_score FROM medication_risk WHERE person_id = ?",
                (person_id,),
            ).fetchone()
        return int(row["risk_score"]) if row else 0

    def get_recent_events(self, person_id: str, days: int = 30) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM medication_event
                WHERE person_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (person_id, cutoff_iso),
            ).fetchall()
        return [dict(r) for r in rows]

    def _find_matching_schedule_for_event(
        self, person_id: str, med_name: str, ts: datetime
    ) -> Optional[Dict]:
        """
        Very small heuristic: same-day schedule with same name where the
        event time lies within the tolerance window and has not yet been
        matched by any other event.
        """
        today_str = ts.strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            schedules = conn.execute(
                """
                SELECT *
                FROM medication_schedule
                WHERE person_id = ?
                  AND medication_name = ?
                """,
                (person_id, med_name),
            ).fetchall()

            for s in schedules:
                sched_time = _parse_hhmm(s["time_of_day"])
                sched_dt = datetime.combine(date.fromisoformat(today_str), sched_time)
                tol = timedelta(minutes=int(s["tolerance_min"]))
                window_start = sched_dt - tol
                window_end = sched_dt + tol
                if not (window_start <= ts <= window_end):
                    continue

                # Already matched?
                used = conn.execute(
                    """
                    SELECT 1
                    FROM medication_event
                    WHERE scheduled_id = ?
                    LIMIT 1
                    """,
                    (s["id"],),
                ).fetchone()
                if used:
                    continue
                return dict(s)
        return None

    def record_event(
        self,
        person_id: str,
        med_name: str,
        ts: datetime,
        source: str = "ai",
    ) -> Dict:
        ts_iso = ts.isoformat()
        schedule = self._find_matching_schedule_for_event(person_id, med_name, ts)
        scheduled_id = schedule["id"] if schedule else None

        on_time = 1
        if schedule:
            sched_time = _parse_hhmm(schedule["time_of_day"])
            sched_dt = datetime.combine(ts.date(), sched_time)
            tol = timedelta(minutes=int(schedule["tolerance_min"]))
            if ts > sched_dt + tol:
                on_time = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO medication_event
                    (person_id, medication_name, timestamp, scheduled_id, on_time, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (person_id, med_name, ts_iso, scheduled_id, on_time, source),
            )
            conn.commit()

        # Simple risk-update heuristic
        current_risk = self._get_or_init_risk(person_id)
        if schedule:
            if on_time:
                new_risk = max(0, current_risk - 5)
            else:
                new_risk = min(100, current_risk + 10)
        else:
            new_risk = current_risk

        new_risk = self._set_risk(person_id, new_risk)

        return {
            "person_id": person_id,
            "medication_name": med_name,
            "timestamp": ts_iso,
            "scheduled_id": scheduled_id,
            "on_time": bool(on_time),
            "new_risk_score": new_risk,
        }

    # ── Reminder / missed-dose detection ───────────────────────────────────
    def check_and_trigger_reminders(self, person_id: str, speaker=None) -> List[Dict]:
        """
        Check all schedules for today. For any schedule whose window has fully
        passed without a matching event, create a 'missed' event, bump risk,
        and (optionally) trigger a TTS reminder via `speaker(text)`.

        Returns a list of dicts describing which schedules triggered.
        """
        now = datetime.utcnow()
        today = date.today()
        triggered: List[Dict] = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            schedules = conn.execute(
                """
                SELECT *
                FROM medication_schedule
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchall()

            for s in schedules:
                sched_time = _parse_hhmm(s["time_of_day"])
                sched_dt = datetime.combine(today, sched_time)
                tol = timedelta(minutes=int(s["tolerance_min"]))
                window_end = sched_dt + tol

                # Only consider past schedules whose tolerance window has ended
                if now <= window_end:
                    continue

                # Already has any event (taken or missed)?
                existing = conn.execute(
                    """
                    SELECT 1
                    FROM medication_event
                    WHERE person_id = ?
                      AND scheduled_id = ?
                    LIMIT 1
                    """,
                    (person_id, s["id"]),
                ).fetchone()
                if existing:
                    continue

                # Mark as missed
                missed_ts = window_end
                conn.execute(
                    """
                    INSERT INTO medication_event
                        (person_id, medication_name, timestamp, scheduled_id, on_time, source)
                    VALUES (?, ?, ?, ?, 0, 'missed')
                    """,
                    (
                        person_id,
                        s["medication_name"],
                        missed_ts.isoformat(),
                        s["id"],
                    ),
                )
                conn.commit()

                # Risk bump for a fully missed dose
                current_risk = self._get_or_init_risk(person_id)
                new_risk = self._set_risk(person_id, min(100, current_risk + 15))

                message = (
                    f"It is time to take your medicine: {s['medication_name']}. "
                    "Please take your medication now."
                )
                if speaker is not None:
                    try:
                        speaker(message)
                    except Exception:
                        # Silently ignore TTS errors; this is best-effort.
                        pass

                triggered.append(
                    {
                        "schedule_id": s["id"],
                        "medication_name": s["medication_name"],
                        "new_risk_score": new_risk,
                    }
                )

        return triggered

