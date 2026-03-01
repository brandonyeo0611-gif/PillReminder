"""
dashboard.py
=============
The family-facing Streamlit dashboard.
This is the most important file for the hackathon demo.

Shows:
  1. Live activity status (most recent prediction)
  2. Today's activity timeline vs baseline
  3. Risk score with anomaly breakdown
  4. 7-day history calendar

USAGE:
  pip3 install streamlit plotly
  streamlit run app/dashboard.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import datetime, timedelta
import time

from src.logger import ActivityLogger
from src.baseline_builder import BaselineBuilder
from src.deviation_detector import DeviationDetector, _hour_to_str

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareWatch",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');
    .main { background: #080b12; font-family: 'IBM Plex Mono', 'Courier New', monospace; }
    .metric-card {
        background: #0d1117;
        border: 1px solid #1e2535;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .risk-green  { color: #00e676; font-size: 3rem; font-weight: bold; }
    .risk-yellow { color: #ffeb3b; font-size: 3rem; font-weight: bold; }
    .risk-red    { color: #ff1744; font-size: 3rem; font-weight: bold; }
    .activity-label {
        font-size: 1.8rem;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        display: inline-block;
    }
    .stMetric { background: #0d1117; border: 1px solid #1e2535; border-radius: 10px; padding: 10px; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #2d3550; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── ACTIVITY COLOURS (matches React component: color, icon, label) ──────────────
ACTIVITIES = {
    "sitting":     {"color": "#4a9eff", "icon": "🪑", "label": "Sitting"},
    "eating":      {"color": "#00e676", "icon": "🍽️", "label": "Eating"},
    "walking":     {"color": "#ff9800", "icon": "🚶", "label": "Walking"},
    "pill_taking": {"color": "#ffeb3b", "icon": "💊", "label": "Pill Taking"},
    "lying_down":  {"color": "#ce93d8", "icon": "🛋️", "label": "Lying Down"},
    "no_person":   {"color": "#546e7a", "icon": "👤", "label": "Away"},
    "unknown":     {"color": "#8892a4", "icon": "❓", "label": "Unknown"},
    "fallen":      {"color": "#ff1744", "icon": "⚠️", "label": "FALLEN"},
}
# Legacy alias for code that still uses COLOURS (e.g. timeline)
COLOURS = {k: v["color"] for k, v in ACTIVITIES.items()}

PERSON_ID = "resident"  # change to support multiple residents later

# ── INIT ──────────────────────────────────────────────────────────────────────
logger    = ActivityLogger()
builder   = BaselineBuilder(logger)
detector  = DeviationDetector()

# ── DUMMY DATA (used when no real logs exist yet) ──────────────────────────────
def inject_demo_data():
    """Inject fake but realistic data so the dashboard looks great in demo mode."""
    import random
    from datetime import datetime, timedelta

    activities = ["sitting","eating","walking","pill_taking","lying_down","sitting","walking","eating"]
    base = datetime.now().replace(hour=7, minute=0, second=0)

    for day_offset in range(7):
        for i, act in enumerate(activities):
            t = base - timedelta(days=day_offset) + timedelta(hours=i*1.5)
            # small random jitter
            t += timedelta(minutes=random.randint(-20, 20))
            entry_time = t.strftime("%Y-%m-%dT%H:%M:%S")
            with __import__("sqlite3").connect(logger.db_path) as conn:
                conn.execute("""
                    INSERT INTO activity_log
                        (person_id, timestamp, date, hour, minute, activity, confidence)
                    VALUES (?,?,?,?,?,?,?)
                """, (PERSON_ID, entry_time, t.strftime("%Y-%m-%d"),
                      t.hour, t.minute, act, round(random.uniform(0.75, 0.98), 2)))
                conn.commit()
    print("✅ Demo data injected")


def logs_to_timeline(logs: list[dict]) -> list[dict]:
    """Convert activity_log rows to [{time, activity, conf, note}] for React-style display."""
    out = []
    for row in logs:
        ts = row.get("timestamp", "")
        t = ts[11:16] if len(ts) >= 16 else f"{row.get('hour', 0):02d}:{row.get('minute', 0):02d}"
        out.append({
            "time": t,
            "activity": row.get("activity", "unknown"),
            "conf": row.get("confidence", 0.9),
            "note": "",
        })
    return out


def build_week_data(all_logs: list[dict], risk_result: dict) -> list[dict]:
    """Build 7-day [{day, risk, pill, events}] from get_last_n_days output.
    Today uses detector risk_score; other days use heuristic from events + pill compliance.
    """
    from datetime import datetime, timedelta
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df = pd.DataFrame(all_logs) if all_logs else pd.DataFrame()
    base = datetime.now().date()
    out = []
    for i in range(7):
        d = base - timedelta(days=6 - i)
        day_name = days[d.weekday()]
        day_str = d.strftime("%Y-%m-%d")
        day_logs = df[df["date"] == day_str] if not df.empty and len(df.columns) > 0 and "date" in df.columns else pd.DataFrame()
        events = len(day_logs)
        pill = bool((day_logs["activity"] == "pill_taking").any()) if not day_logs.empty and "activity" in day_logs.columns else False
        risk = risk_result.get("risk_score", 15) if i == 6 else min(90, 10 + events + (0 if pill else 25))
        out.append({"day": day_name, "risk": risk, "pill": pill, "events": int(events)})
    return out


def render_risk_gauge(score: int):
    """Plotly gauge matching React RiskGauge: 0-100 arc, color by zone (green/yellow/red)."""
    color = "#00e676" if score <= 30 else "#ffeb3b" if score <= 60 else "#ff1744"
    label = "NORMAL" if score <= 30 else "ELEVATED" if score <= 60 else "CRITICAL"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "#0d1117",
            "borderwidth": 2,
            "bordercolor": "#1e2535",
            "threshold": {"line": {"color": "#ff1744", "width": 4}, "thickness": 0.75, "value": 60},
            "steps": [
                {"range": [0, 30], "color": "#00e676"},
                {"range": [30, 60], "color": "#ffeb3b"},
                {"range": [60, 100], "color": "#ff1744"},
            ],
        },
        title={"text": label, "font": {"size": 12}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=200, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def render_activity_feed(timeline: list[dict], limit: int = 8):
    """Render Activity Log as React-style cards: icon, label, time, confidence. Newest first."""
    for item in list(reversed(timeline))[:limit]:
        act = ACTIVITIES.get(item["activity"], ACTIVITIES["unknown"])
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;
                   background:#0d1117;border:1px solid #1e2535;margin:4px 0;">
            <span style="font-size:16px">{act['icon']}</span>
            <div>
                <div style="color:{act['color']};font-size:12px;font-weight:700">{act['label'].upper()}</div>
                <div style="color:#4a5568;font-size:10px">{item['time']} · {int(item['conf']*100)}% confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_week_calendar(week_data: list[dict]):
    """Render 7-day calendar: day name, risk bar (height=risk%), pill icon. Matches React WeekCalendar."""
    cols = st.columns(7)
    for i, d in enumerate(week_data):
        with cols[i]:
            color = "#00e676" if d["risk"] <= 30 else "#ffeb3b" if d["risk"] <= 60 else "#ff1744"
            st.markdown(f"""
            <div style="text-align:center;padding:10px;border-radius:10px;background:#1a1f2e;">
                <div style="font-size:9px;color:#4a5568;letter-spacing:1px">{d['day']}</div>
                <div style="height:40px;background:#0d1117;border-radius:4px;position:relative;margin:4px 0;overflow:hidden;">
                    <div style="position:absolute;bottom:0;left:0;right:0;height:{d['risk']}%;background:{color};border-radius:4px;"></div>
                </div>
                <div style="font-size:10px;color:{color};font-weight:700">{d['risk']}</div>
                <div style="font-size:14px">{'💊' if d['pill'] else '❌'}</div>
            </div>
            """, unsafe_allow_html=True)


def get_medication_schedule(baseline: dict | None, today_logs: list[dict], demo_mode: str) -> list[dict]:
    """Build medication schedule [{time, label, done, actual}] from today's logs.
    Crisis mode overrides to show missed morning dose.
    """
    schedule = [
        {"time": "08:00", "label": "Morning dose", "done": False, "actual": None},
        {"time": "13:00", "label": "Lunch dose", "done": False, "actual": None},
        {"time": "21:00", "label": "Night dose", "done": False, "actual": None},
    ]
    pill_logs = [l for l in today_logs if l.get("activity") == "pill_taking"]
    for i, med in enumerate(schedule):
        for p in pill_logs:
            h, m = p.get("hour", 0), p.get("minute", 0)
            pt = f"{h:02d}:{m:02d}"
            if i == 0 and h < 12:
                med["done"] = True
                med["actual"] = pt
            elif i == 1 and 11 <= h <= 14:
                med["done"] = True
                med["actual"] = pt
            elif i == 2 and h >= 20:
                med["done"] = True
                med["actual"] = pt
    if demo_mode == "crisis":
        schedule[0]["done"] = False
        schedule[0]["actual"] = None
    return schedule


# ── SESSION STATE ────────────────────────────────────────────────────────────
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = "normal"
if "alert_ack" not in st.session_state:
    st.session_state.alert_ack = False

# ── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_title, col_btns, col_live, col_time = st.columns([1, 4, 2, 1, 1])
with col_logo:
    st.markdown("""
    <div style="width:36px;height:36px;border-radius:8px;background:linear-gradient(135deg,#4a9eff,#0057ff);
                display:flex;align-items:center;justify-content:center;font-size:18px;">👁️</div>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown("# CareWatch")
    st.caption("INTELLIGENT ELDERLY MONITORING")
with col_btns:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("NORMAL DAY", key="norm"):
            st.session_state.demo_mode = "normal"
            st.rerun()
    with b2:
        if st.button("CRISIS MODE", key="crisis"):
            st.session_state.demo_mode = "crisis"
            st.rerun()
with col_live:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:6px;padding:5px 12px;background:#0d1117;border-radius:20px;">
        <div style="width:6px;height:6px;border-radius:50%;background:#00e676;"></div>
        <span style="font-size:10px;color:#00e676;letter-spacing:2px">LIVE</span>
    </div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"<div style='font-size:11px;color:#4a5568'>{datetime.now().strftime('%I:%M %p')}</div>", unsafe_allow_html=True)

# ── CHECK FOR DATA, INJECT DEMO IF EMPTY ──────────────────────────────────────
today_logs = logger.get_today(PERSON_ID)
if not today_logs:
    st.info("No live data yet — showing demo mode. Run realtime_inference.py to see real data.")
    if st.button("Load demo data"):
        inject_demo_data()
        builder.build_baseline(PERSON_ID)
        st.rerun()

# ── DATA FETCH ────────────────────────────────────────────────────────────────
risk_result = detector.check(PERSON_ID)
baseline = builder.load_baseline(PERSON_ID)
last_log = logger.get_last_activity(PERSON_ID)
all_logs = logger.get_last_n_days(7, PERSON_ID)

demo_mode = st.session_state.demo_mode
display_risk = 73 if demo_mode == "crisis" else risk_result["risk_score"]
timeline = logs_to_timeline(today_logs) if today_logs else []
week_data = build_week_data(all_logs or [], {"risk_score": display_risk})
med_schedule = get_medication_schedule(baseline, today_logs or [], demo_mode)

# ── ALERT BANNER (Crisis only) ────────────────────────────────────────────────
if demo_mode == "crisis" and not st.session_state.alert_ack:
    alert_col1, alert_col2 = st.columns([4, 1])
    with alert_col1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a0a0a,#2d0f0f);border:1px solid #ff1744;border-radius:10px;
                    padding:12px 20px;box-shadow:0 0 30px rgba(255,23,68,0.2);">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="font-size:20px">🚨</div>
                <div>
                    <div style="color:#ff6b6b;font-weight:700;font-size:13px;letter-spacing:1px">CRITICAL ALERT — MRS TAN</div>
                    <div style="color:#ff9999;font-size:11px;margin-top:2px">Pill taking not detected · Expected 8:45am · Now 2h 15m overdue</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with alert_col2:
        if st.button("ACKNOWLEDGE"):
            st.session_state.alert_ack = True
            st.rerun()
else:
    st.session_state.alert_ack = False

# ── ROW 1: Current Activity | RiskGauge | Medication (3-col) ────────────────────
col_act, col_risk, col_med = st.columns(3)
current_act = last_log["activity"] if last_log else "unknown"
act_info = ACTIVITIES.get(current_act, ACTIVITIES["unknown"])

with col_act:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{act_info['color']}15,#0d1117);border:1px solid {act_info['color']}30;
                border-radius:14px;padding:20px;">
        <div style="font-size:9px;color:#4a5568;letter-spacing:3px">CURRENT ACTIVITY</div>
        <div style="display:flex;align-items:center;gap:14px;margin-top:8px;">
            <span style="font-size:32px">{act_info['icon']}</span>
            <div>
                <div style="font-size:22px;font-weight:800;color:{act_info['color']}">{act_info['label'].upper()}</div>
                <div style="color:#4a5568;font-size:10px">{int((last_log or {}).get('confidence',0.9)*100)}% CONFIDENCE</div>
            </div>
        </div>
        <div style="background:#0d1117;border-radius:8px;padding:8px 12px;margin-top:8px;font-size:10px;color:#4a5568">
            <span style="color:#8892a4">RESIDENT</span> Mrs Tan, 74<br>
            <span style="color:#8892a4">LAST SEEN</span> {datetime.now().strftime('%H:%M')} · Living Room
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_risk:
    st.markdown("<div style='font-size:9px;color:#4a5568;letter-spacing:3px'>DAILY RISK SCORE</div>", unsafe_allow_html=True)
    st.plotly_chart(render_risk_gauge(display_risk), use_container_width=True)
    if demo_mode == "crisis":
        for line in ["🔴 Pill not taken +40pts", "🟡 Inactivity 3h+ +25pts", "🔵 Late breakfast +8pts"]:
            st.markdown(f"<div style='font-size:10px;margin:2px 0'>{line}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ All activities within normal range")

with col_med:
    st.markdown("<div style='font-size:9px;color:#4a5568;letter-spacing:3px'>MEDICATION</div>", unsafe_allow_html=True)
    for med in med_schedule:
        border_c = "#ff174430" if not med["done"] and demo_mode == "crisis" else "#2d3550"
        bg_c = "#ff174408" if not med["done"] and demo_mode == "crisis" else "#00e67608" if med["done"] else "#1a1f2e"
        icon = "❌" if not med["done"] and demo_mode == "crisis" else "✅" if med["done"] else "⏳"
        actual_str = f" → Taken {med['actual']}" if med["actual"] else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;
                    background:{bg_c};border:1px solid {border_c};margin:4px 0;">
            <span style="font-size:16px">{icon}</span>
            <div>
                <div style="font-size:11px;color:#8892a4">{med['label']}</div>
                <div style="font-size:10px;color:#4a5568">Scheduled {med['time']}{actual_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── ROW 2: TIMELINE ───────────────────────────────────────────────────────────
st.subheader("📅 Today's Activity Timeline")

if today_logs:
    df = pd.DataFrame(today_logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["end_time"]  = df["timestamp"].shift(-1).fillna(df["timestamp"] + pd.Timedelta(minutes=15))

    fig = go.Figure()

    for _, row in df.iterrows():
        colour = COLOURS.get(row["activity"], "#757575")
        label  = row["activity"].replace("_"," ").title()
        fig.add_trace(go.Bar(
            x=[(row["end_time"] - row["timestamp"]).seconds / 3600],
            y=["Today"],
            base=[(row["timestamp"] - row["timestamp"].replace(hour=0,minute=0,second=0)).seconds / 3600],
            orientation="h",
            marker_color=colour,
            name=label,
            hovertemplate=f"<b>{label}</b><br>{row['timestamp'].strftime('%I:%M %p')}<br>Confidence: {row['confidence']*100:.0f}%<extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        barmode="stack",
        height=200,
        paper_bgcolor="#1a1f2e",
        plot_bgcolor="#1a1f2e",
        font_color="white",
        xaxis=dict(title="Hour of Day", range=[6, 22], tickvals=list(range(6,23)),
                   ticktext=[f"{h}{'am' if h<12 else 'pm'}" for h in range(6,23)],
                   gridcolor="#2d2d2d"),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No activity data for today yet.")

# ── ROW 3: Activity Log | 7-Day History ───────────────────────────────────────
col_feed, col_week = st.columns([1.4, 1])

with col_feed:
    st.markdown("<div style='font-size:9px;color:#4a5568;letter-spacing:3px;margin-bottom:12px'>ACTIVITY LOG</div>", unsafe_allow_html=True)
    if timeline:
        render_activity_feed(timeline)
    else:
        st.info("No activity log yet.")

with col_week:
    st.markdown("<div style='font-size:9px;color:#4a5568;letter-spacing:3px;margin-bottom:12px'>7-DAY HISTORY · 💊 = pill taken</div>", unsafe_allow_html=True)
    if week_data:
        render_week_calendar(week_data)
        pill_compliance = sum(1 for d in week_data if d["pill"]) / 7 * 100 if week_data else 0
        st.markdown(f"""
        <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
            <div style="background:#1a1f2e;border-radius:8px;padding:8px 10px;text-align:center">
                <div style="font-size:16px;font-weight:800;color:#ffeb3b">{pill_compliance:.0f}%</div>
                <div style="font-size:9px;color:#4a5568">Pill compliance</div>
            </div>
            <div style="background:#1a1f2e;border-radius:8px;padding:8px 10px;text-align:center">
                <div style="font-size:16px;font-weight:800;color:#4a9eff">{display_risk}</div>
                <div style="font-size:9px;color:#4a5568">Avg risk score</div>
            </div>
            <div style="background:#1a1f2e;border-radius:8px;padding:8px 10px;text-align:center">
                <div style="font-size:16px;font-weight:800;color:#00e676">{sum(1 for d in week_data if d['events']>0)}/7</div>
                <div style="font-size:9px;color:#4a5568">Active days</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Not enough history yet.")

# Baseline profile (collapsible)
with st.expander("📊 Baseline Profile"):
    if baseline and baseline.get("activities"):
        rows = []
        for act, stats in baseline["activities"].items():
            if stats["mean_hour"] is not None:
                rows.append({
                    "Activity":      act.replace("_"," ").title(),
                    "Usual Time":    _hour_to_str(stats["mean_hour"]),
                    "Avg per Day":   f"{stats['mean_count']:.1f}x",
                    "Daily?":        "✅" if stats["occurs_daily"] else "❌",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Baseline not built yet — needs 7 days of data.")
        if st.button("Build baseline from existing logs"):
            builder.build_baseline(PERSON_ID)
            st.rerun()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("CareWatch — Early detection of routine deviation in elderly individuals. "
           "Built for the Singapore Innovation Challenge 2026.")

# Auto-refresh every 30 seconds
time.sleep(0.1)
st.markdown(
    '<meta http-equiv="refresh" content="30">',
    unsafe_allow_html=True
)