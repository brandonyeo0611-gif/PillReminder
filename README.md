# CareWatch 👁️
> AI-powered elderly routine monitoring. 
> Detects behavioural decline before crisis happens.

## The Problem
1 in 3 elderly Singaporeans live alone. 
Families only find out something is wrong after a fall or hospitalisation.
Health decline shows up in behaviour days before a crisis.

## The Solution
CareWatch learns a resident's normal daily routine over 7 days.
It flags when something feels wrong — before the crisis.

## System Architecture
CAMERA FEED
     ↓
┌─────────────────────────────────────────┐
│           PERCEPTION ENGINE             │
│  YOLO11x-pose → 17 keypoints/frame      │
│  + Person tracking (ByteTrack)          │  
│  + Confidence filtering (>0.6)          │  
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         FEATURE ENGINEERING             │
│  Raw keypoints → 8 joint angles         │
│  + Velocity (how fast angles change)    │  
│  + Symmetry score (left vs right)       │  
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         ACTIVITY CLASSIFIER             │
│  AngleLSTMNet (sequence of 30 frames)   │
│  → sitting / eating / walking /         │
│     pill_taking / lying / no_person     │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│           MEMORY ENGINE                 │
│  SQLite: logs every prediction          │
│  {person_id, timestamp, activity,       │
│   confidence, angles, velocity}         │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│        INTELLIGENCE ENGINE              │
│  baseline_builder: 7-day profile        │
│  deviation_detector: z-score anomaly    │
│  risk_scorer: weighted urgency 0-100    │  
└─────────────────┬───────────────────────┘
                  ↓
          ┌───────┴────────┐
          ↓                ↓
┌──────────────┐   ┌──────────────────┐
│ ALERT ENGINE │   │    DASHBOARD     │  ← THIS IS THE DEMO
│ Telegram bot │   │   Streamlit app  │
│ SMS fallback │   │   (family view)  │
└──────────────┘   └──────────────────┘

## What It Detects
- Missed medication
- Unusual inactivity (3+ hours)
- Routine deviation (eating/walking at wrong times)
- Falls (immediate alert)

## Alert Example
🚨 Mrs Tan has not taken her medication.
No movement detected since 9:10am.
Pill expected at 8:45am — now 2h 15m overdue.

## Tech Stack
- YOLO11x-pose — real-time pose estimation
- PyTorch LSTM — activity classification
- SQLite — activity logging
- Streamlit — family dashboard
- Telegram Bot — instant family alerts

## Setup
pip install -r requirements.txt
python3 app/realtime_inference.py   # live demo
streamlit run app/dashboard.py      # dashboard

## Team
Built for Singapore Innovation Challenge 2026