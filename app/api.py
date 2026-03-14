"""
api.py
======
FastAPI REST API for CareWatch. Exposes logger, baseline, and deviation detector
to the React frontend. Runs alongside realtime_inference.py (which writes to the
same SQLite DB). Use --workers 1 (SQLite does not handle concurrent writes).

USAGE:
  uvicorn app.api:app --reload --port 8000 --workers 1
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.logger import ActivityLogger
from src.baseline_builder import BaselineBuilder
from src.deviation_detector import DeviationDetector
from src.medication import MedicationRepo
from src.medication_ai import MedicationAI
from src.label_detector import MedicationLabelDetector
from src.tts import speak

app = FastAPI(title="CareWatch API")

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = ActivityLogger()
builder = BaselineBuilder(logger)
detector = DeviationDetector()
med_repo = MedicationRepo()
med_ai = MedicationAI()
label_detector = MedicationLabelDetector()
PERSON = "resident"


class MedicationScheduleIn(BaseModel):
    medication_name: str
    dose: Optional[str] = None
    time_of_day: str          # "HH:MM"
    tolerance_min: int = 30
    illness_hint: Optional[str] = None


class MedicationScheduleOut(MedicationScheduleIn):
    id: int


class MedicationEventIn(BaseModel):
    medication_name: str
    detected_at: Optional[datetime] = None
    source: str = "ai"  # "ai" or "manual"


def _inject_demo_data():
    """Full implementation from dashboard.py — inserts 7 days of fake activity logs."""
    activities = [
        "sitting", "eating", "walking", "pill_taking",
        "lying_down", "sitting", "walking", "eating",
    ]
    base = datetime.now().replace(hour=7, minute=0, second=0, microsecond=0)
    with sqlite3.connect(logger.db_path) as conn:
        for day_offset in range(7):
            for i, act in enumerate(activities):
                t = base - timedelta(days=day_offset) + timedelta(hours=i * 1.5)
                t += timedelta(minutes=random.randint(-20, 20))
                conn.execute(
                    """
                    INSERT INTO activity_log
                        (person_id, timestamp, date, hour, minute, activity, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        PERSON,
                        t.strftime("%Y-%m-%dT%H:%M:%S"),
                        t.strftime("%Y-%m-%d"),
                        t.hour,
                        t.minute,
                        act,
                        round(random.uniform(0.75, 0.98), 2),
                    ),
                )
        conn.commit()


@app.get("/api/logs/today")
def get_today():
    """Return all logs for today. Used by timeline and medication schedule."""
    return logger.get_today(PERSON) or []


@app.get("/api/logs/latest")
def get_latest():
    """Return the most recent activity log. Used for current activity display."""
    return logger.get_last_activity(PERSON) or {}


@app.get("/api/logs/week")
def get_week():
    """Return logs for the last 7 days. Used for week history and buildWeekData."""
    return logger.get_last_n_days(7, PERSON) or []


@app.get("/api/risk")
def get_risk():
    """Return combined behavioural + medication adherence risk."""
    base = detector.check(PERSON)
    med_risk = med_repo.get_medication_risk(PERSON)
    combined = dict(base)
    combined_score = max(0, min(100, base.get("risk_score", 0) + med_risk))
    combined["risk_score"] = combined_score
    combined["medication_risk_component"] = med_risk
    return combined


@app.get("/api/baseline")
def get_baseline():
    """Return baseline profile. baseline_risk is placeholder until risk history stored."""
    baseline = builder.load_baseline(PERSON) or {}
    baseline["baseline_risk"] = 15
    return baseline


@app.post("/api/baseline/build")
def build_baseline_endpoint():
    """Build baseline from existing logs. Call after inject or when new data available."""
    builder.build_baseline(PERSON)
    return {"ok": True}


@app.post("/api/demo/inject")
def inject_demo():
    """Inject 7 days of demo data and build baseline. For demos when no live data."""
    _inject_demo_data()
    builder.build_baseline(PERSON)
    return {"ok": True, "message": "Demo data injected and baseline built"}


@app.get("/api/medication/schedules", response_model=List[MedicationScheduleOut])
def list_schedules():
    """Return all medication schedules for the resident."""
    return med_repo.list_schedules(PERSON)


@app.post("/api/medication/schedules", response_model=MedicationScheduleOut)
def create_schedule(payload: MedicationScheduleIn):
    """Create a new medication schedule entry."""
    return med_repo.create_schedule(PERSON, payload)


@app.delete("/api/medication/schedules/{schedule_id}")
def delete_schedule(schedule_id: int):
    """Delete a medication schedule entry."""
    med_repo.delete_schedule(PERSON, schedule_id)
    return {"ok": True}


@app.post("/api/medication/event")
def record_medication_event(payload: MedicationEventIn):
    """
    Record a detected medication intake event from the AI webcam or manual input.
    Updates the medication risk component based on timeliness.
    """
    ts = payload.detected_at or datetime.utcnow()
    result = med_repo.record_event(PERSON, payload.medication_name, ts, payload.source)
    return result


@app.post("/api/medication/check-reminders")
def check_reminders():
    """
    Check all schedules and trigger verbal reminders (text-to-speech)
    for any doses that have passed their tolerance window without detection.
    """
    triggered = med_repo.check_and_trigger_reminders(PERSON, speaker=speak)
    return {"ok": True, "triggered": triggered}


@app.get("/api/medication/recommendations")
def get_medication_recommendations():
    """
    Guess likely illnesses based on recent medication history and return them.
    Frontend can map these illnesses to diet and lifestyle suggestions.
    """
    events = med_repo.get_recent_events(PERSON, days=30)
    illnesses = med_ai.guess_illnesses(events)
    return {"illnesses": illnesses}

@app.post("/api/medication/scan")
async def scan_medication_label(file: UploadFile = File(...)):
    """
    Accepts an uploaded image of a medication label.
    Uses MedicationLabelDetector to extract name and dose.
    """
    contents = await file.read()
    result = label_detector.extract_from_image(contents)
    return result

