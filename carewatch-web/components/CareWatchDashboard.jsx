"use client";

import { useState, useEffect } from "react";

// ── Constants ─────────────────────────────────────────────────────────────────

const ACTIVITIES = {
  sitting:     { color: "#4a9eff", label: "Sitting",     icon: "SIT" },
  eating:      { color: "#00e676", label: "Eating",      icon: "EAT" },
  walking:     { color: "#ff9800", label: "Walking",     icon: "WLK" },
  pill_taking: { color: "#ffeb3b", label: "Pill Taking", icon: "MED" },
  lying_down:  { color: "#ce93d8", label: "Lying Down",  icon: "LIE" },
  no_person:   { color: "#546e7a", label: "Away",        icon: "---" },
  unknown:     { color: "#8892a4", label: "Unknown",     icon: "???" },
  fallen:      { color: "#ff1744", label: "FALLEN",      icon: "!!!" },
};

const FOOD_RECOMMENDATIONS = {
  "Diabetes": [
    "Avoid sugary foods and sweetened drinks",
    "Choose whole grains over refined carbohydrates",
    "Eat lean protein (chicken, fish, tofu)",
    "Include non-starchy vegetables daily"
  ],
  "Hypertension (High BP)": [
    "Reduce sodium/salt intake (< 1500mg/day)",
    "Eat potassium-rich foods (bananas, potatoes)",
    "Avoid processed and canned foods",
    "Limit alcohol consumption"
  ],
  "High Cholesterol": [
    "Limit saturated fats (red meat, full-fat dairy)",
    "Avoid trans fats (fried foods, baked goods)",
    "Eat high-fiber foods (oats, beans, berries)",
    "Choose healthy fats (olive oil, nuts, avocados)"
  ],
  "Gout": [
    "Limit purine-rich foods (red meat, organ meats)",
    "Avoid alcohol, especially beer",
    "Drink plenty of water to flush uric acid",
    "Eat vitamin C-rich foods"
  ],
  "Acid Reflux (GERD)": [
    "Avoid spicy, acidic, and tomato-based foods",
    "Limit caffeine and chocolate",
    "Eat smaller, more frequent meals",
    "Don't eat within 3 hours of bedtime"
  ],
  "Kidney Disease": [
    "Limit phosphorus-rich foods (dairy, colas)",
    "Restrict potassium depending on stage",
    "Consume high-quality protein in moderation",
    "Control sodium intake strictly"
  ],
  "Lactose Intolerance": [
    "Avoid regular dairy (milk, cheese, ice cream)",
    "Choose lactose-free alternatives",
    "Check labels for hidden whey or casein"
  ],
  "Celiac Disease": [
    "Strictly avoid gluten (wheat, barley, rye)",
    "Choose naturally gluten-free grains (quinoa, rice)"
  ]
};

const LIFESTYLE_RECOMMENDATIONS = {
  "Diabetes": [
    "Maintain 150 mins of moderate aerobic activity weekly",
    "Monitor blood sugar regularly, especially around exercise",
    "Check feet daily for cuts or sores",
    "Stay hydrated, especially during warmer days"
  ],
  "Hypertension (High BP)": [
    "Engage in daily brisk walking or swimming (30 mins)",
    "Practice stress reduction techniques (meditation, deep breathing)",
    "Ensure 7-8 hours of quality sleep",
    "Avoid exercising during extreme heat"
  ],
  "High Cholesterol": [
    "Aim for 30 minutes of cardiovascular exercise daily",
    "Incorporate resistance training 2 times a week",
    "Avoid smoking or secondhand smoke exposure"
  ],
  "Gout": [
    "Keep relevant joints elevated during painful flare-ups",
    "Maintain a healthy body weight gradually (avoid crash diets)",
    "Engage in low-impact exercises like cycling or swimming protect joints"
  ],
  "Acid Reflux (GERD)": [
    "Wait at least 2 hours after meals before exercising",
    "Elevate the head of your bed by 6-8 inches",
    "Wear loose-fitting clothing around the abdomen",
    "Try light walking after meals to aid digestion"
  ],
  "Kidney Disease": [
    "Consult your doctor before starting new exercise routines",
    "Monitor blood pressure daily at home",
    "Pace activities to avoid extreme fatigue"
  ],
  "Lactose Intolerance": [
    "Consider taking a lactase enzyme before occasional dairy consumption",
    "Ensure adequate calcium/Vitamin D intake through supplements or sunlight"
  ],
  "Celiac Disease": [
    "Use separate toasters and cutting boards to avoid cross-contamination",
    "Read medication and supplement labels carefully for hidden gluten"
  ]
};

const DEMO_NORMAL = {
  current:     "eating",
  confidence:  0.91,
  risk:        18,
  baselineRisk:15,
  apiOk:       false,
  timeline: [
    { hour: 7.0,  activity: "lying_down",  duration: 0.5  },
    { hour: 7.5,  activity: "walking",     duration: 0.5  },
    { hour: 8.0,  activity: "pill_taking", duration: 0.25 },
    { hour: 8.25, activity: "eating",      duration: 0.75 },
    { hour: 9.0,  activity: "sitting",     duration: 2.0  },
    { hour: 11.0, activity: "walking",     duration: 0.5  },
    { hour: 11.5, activity: "eating",      duration: 0.75 },
    { hour: 12.25,activity: "sitting",     duration: 1.75 },
    { hour: 14.0, activity: "lying_down",  duration: 1.0  },
    { hour: 15.0, activity: "sitting",     duration: 1.0  },
    { hour: 16.0, activity: "eating",      duration: 0.5  },
  ],
  alerts: [
    { time: "+0.0s", label: "MORNING PILL",    sub: "Pill taken at 08:05 · on schedule",     color: "#00e676", severity: "ok"   },
    { time: "+0.2s", label: "BREAKFAST",        sub: "Eating detected 08:25 · normal window", color: "#00e676", severity: "ok"   },
    { time: "+0.5s", label: "ACTIVITY NORMAL",  sub: "Walking 11:00 · within baseline",       color: "#4a9eff", severity: "info" },
    { time: "+1.1s", label: "NAP DETECTED",     sub: "Lying down 14:00 · 60m · normal",       color: "#ce93d8", severity: "info" },
  ],
  week: [
    { day: "MON", risk: 14, pill: true  },
    { day: "TUE", risk: 22, pill: true  },
    { day: "WED", risk: 18, pill: true  },
    { day: "THU", risk: 35, pill: false },
    { day: "FRI", risk: 20, pill: true  },
    { day: "SAT", risk: 12, pill: true  },
    { day: "SUN", risk: 18, pill: true  },
  ],
  medication: [
    { label: "Morning dose", time: "08:00", done: true,  actual: "08:05", requiredFor: "Hypertension (High BP)" },
    { label: "Lunch dose",   time: "13:00", done: true,  actual: "13:12", requiredFor: "Diabetes" },
    { label: "Night dose",   time: "21:00", done: false, actual: null,    requiredFor: "High Cholesterol" },
  ],
  illnesses: ["Diabetes", "Hypertension (High BP)", "High Cholesterol"],
  vitals: {
    temp:      { a: "36.8", unit: "°C"   },
    inRoom:    { a: "YES",  unit: ""      },
  },
};

const DEMO_CRISIS = {
  ...DEMO_NORMAL,
  current:     "lying_down",
  confidence:  0.87,
  risk:        73,
  baselineRisk:15,
  apiOk:       false,
  alerts: [
    { time: "+0.0s", label: "MORNING PILL MISSED", sub: "Expected 08:45 · Now 2h 15m overdue",    color: "#ff1744", severity: "critical" },
    { time: "+0.3s", label: "INACTIVITY DETECTED", sub: "No movement for 3h+ since 09:00",         color: "#ff1744", severity: "critical" },
    { time: "+0.8s", label: "LATE BREAKFAST",       sub: "Eating at 09:40 · 70m late vs baseline", color: "#ffeb3b", severity: "warning"  },
    { time: "+1.2s", label: "TRAVEL SPEED",         sub: "Slow gait vs yesterday · −32%",          color: "#ffeb3b", severity: "warning"  },
    { time: "+1.8s", label: "LONG LIE-DOWN",        sub: "Lying down 12:00 · 2h 10m · unusual",   color: "#ff1744", severity: "critical" },
  ],
  medication: [
    { label: "Morning dose", time: "08:00", done: false, actual: null, requiredFor: "Hypertension (High BP)" },
    { label: "Lunch dose",   time: "13:00", done: false, actual: null, requiredFor: "Diabetes" },
    { label: "Night dose",   time: "21:00", done: false, actual: null, requiredFor: "High Cholesterol" },
  ],
  vitals: {
    temp:      { a: "36.4",unit: "°C"   },
    inRoom:    { a: "YES", unit: ""      },
  },
};

const DEMO_MY_MEDICINES = [
  { id: 1, name: "Metformin 500mg", dosage: "1 tablet", frequency: "2 times/day", duration: "ongoing", times: ["morning", "night"], color: "#4a9eff" },
  { id: 2, name: "Amlodipine 5mg", dosage: "1 tablet", frequency: "1 time/day", duration: "ongoing", times: ["afternoon"], color: "#00e676" },
  { id: 3, name: "Vitamin D 1000IU", dosage: "1 capsule", frequency: "1 time/day", duration: "30 days", times: ["morning"], color: "#ffeb3b" }
];

// Initial live state — same shape as DEMO_NORMAL, apiOk=false means fallback to demo
const INITIAL_LIVE = {
  current:     "unknown",
  confidence:  0,
  risk:        0,
  baselineRisk:15,
  apiOk:       false,
  timeline:    [],
  alerts:      [],
  week:        [],
  medication:  [],
  illnesses:   [],
  vitals: {
    temp:      { a: "---", unit: "°C"   },
    inRoom:    { a: "---", unit: ""      },
  },
};

// ── Pure data helpers ────────────────────────────────────────────────────────

/**
 * Convert activity_log rows to timeline items.
 */
function logsToTimeline(logs) {
  return (logs || []).map((row) => {
    const ts = row.timestamp || "";
    const t =
      ts.length >= 16
        ? ts.slice(11, 16)
        : `${String(row.hour || 0).padStart(2, "0")}:${String(row.minute || 0).padStart(2, "0")}`;
    return {
      time:     t,
      activity: row.activity || "unknown",
      conf:     row.confidence ?? 0.9,
      note:     "",
    };
  });
}

/**
 * Convert flat timeline to segments.
 */
function timelineToRingFormat(timeline) {
  if (!timeline || timeline.length === 0) return DEMO_NORMAL.timeline;
  const segments = [];
  for (const t of timeline) {
    const [h, m] = (t.time || "00:00").split(":").map(Number);
    const hour = h + m / 60;
    const last = segments[segments.length - 1];
    if (last && last.activity === t.activity) {
      last.duration += 0.25;
    } else {
      segments.push({ hour, activity: t.activity, duration: 0.25 });
    }
  }
  return segments;
}

/**
 * Build 7-day week summary.
 */
function buildWeekData(logs, riskResult) {
  const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const base = new Date();
  const out = [];
  for (let i = 0; i < 7; i++) {
    const d = new Date(base);
    d.setDate(d.getDate() - (6 - i));
    const dayStr = d.toISOString().slice(0, 10);
    const dayLogs = (logs || []).filter((r) => r.date === dayStr);
    const events = dayLogs.length;
    const pill = dayLogs.some((r) => r.activity === "pill_taking");
    const risk =
      i === 6
        ? (riskResult?.risk_score ?? 15)
        : Math.min(90, 10 + events + (pill ? 0 : 25));
    out.push({
      day:    DAYS[d.getDay()].toUpperCase(),
      risk,
      pill,
      events,
    });
  }
  return out;
}

/**
 * Build medication schedule from today's logs.
 */
function getMedSchedule(todayLogs, mode) {
  const schedule = [
    { time: "08:00", label: "Morning dose", done: false, actual: null, requiredFor: "General" },
    { time: "13:00", label: "Lunch dose",   done: false, actual: null, requiredFor: "General" },
    { time: "21:00", label: "Night dose",   done: false, actual: null, requiredFor: "General" },
  ];
  const pillLogs = (todayLogs || []).filter((r) => r.activity === "pill_taking");
  for (let i = 0; i < schedule.length; i++) {
    for (const p of pillLogs) {
      const h = p.hour ?? 0;
      const m = p.minute ?? 0;
      const pt = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
      if (i === 0 && h < 12) {
        schedule[i].done = true;
        schedule[i].actual = pt;
      } else if (i === 1 && h >= 11 && h <= 14) {
        schedule[i].done = true;
        schedule[i].actual = pt;
      } else if (i === 2 && h >= 20) {
        schedule[i].done = true;
        schedule[i].actual = pt;
      }
    }
  }
  // Crisis mode override — always show morning dose as missed
  if (mode === "crisis") {
    schedule[0].done   = false;
    schedule[0].actual = null;
  }
  return schedule;
}

// ── Sub-components ───────────────────────────────────────────────────────────

function Spark({ data, color, height = 36 }) {
  const w = 160, h = height;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) =>
      `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * (h - 4) - 2}`
    )
    .join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function RiskScore({ score, label }) {
  const color =
    score <= 30 ? "#00e676" :
    score <= 60 ? "#ffeb3b" :
                  "#ff1744";
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 9, color: "#4a5568", letterSpacing: 3, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 72, fontWeight: 700, color, lineHeight: 1, fontFamily: "'IBM Plex Mono',monospace" }}>{score}</div>
      <div style={{ fontSize: 9, color: "#4a5568", letterSpacing: 2, marginTop: 4 }}>NON-ADHERENCE RISK</div>
    </div>
  );
}

function AlertRow({ alert, revealed }) {
  const dotColor =
    alert.severity === "critical" ? "#ff1744" :
    alert.severity === "warning"  ? "#ffeb3b" :
                                    "#00e676";
  return (
    <div style={{
      opacity:    revealed ? 1 : 0,
      transform:  revealed ? "translateX(0)" : "translateX(10px)",
      transition: "all 0.4s ease",
      borderBottom: "1px solid #1e2535",
      padding: "8px 0",
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: dotColor, marginTop: 3, flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 9, fontWeight: 700, color: alert.color, letterSpacing: 1 }}>{alert.label}</span>
            <span style={{ fontSize: 8, color: "#2d3550" }}>{alert.time}</span>
          </div>
          <div style={{ fontSize: 8, color: "#4a5568", marginTop: 2, lineHeight: 1.4 }}>{alert.sub}</div>
        </div>
      </div>
    </div>
  );
}

function Clock() {
  const [timeStr, setTimeStr] = useState("");
  useEffect(() => {
    const update = () => {
      setTimeStr(new Date().toLocaleTimeString("en-SG", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }));
    };
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, []);
  return <span style={{ fontSize: 10, color: "#2d3550" }}>{timeStr}</span>;
}

// ── Main component ───────────────────────────────────────────────────────────

export default function CareWatchDashboard() {
  const [mode,        setMode]        = useState("normal");
  const [medTab,      setMedTab]      = useState("schedule");
  const [myMedicines, setMyMedicines] = useState(DEMO_MY_MEDICINES);
  const [alertAck,    setAlertAck]    = useState(false);
  const [revealed, setRevealed] = useState([]);
  const [liveData, setLiveData] = useState(INITIAL_LIVE);
  const [injecting,setInjecting]= useState(false);
  
  // Add Medicine Form State
  const [formName, setFormName] = useState("");
  const [formDose, setFormDose] = useState("");
  const [isScanning, setIsScanning] = useState(false);

  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Use live data when API is reachable; otherwise fall back to demo constants
  const data = liveData.apiOk
    ? { ...liveData, mode }
    : mode === "crisis" ? DEMO_CRISIS : DEMO_NORMAL;

  const riskColor = data.risk <= 30 ? "#00e676" : data.risk <= 60 ? "#ffeb3b" : "#ff1744";
  const mono      = "'IBM Plex Mono', 'Courier New', monospace";
  const panel     = { background: "#0d1117", border: "1px solid #1e2535", borderRadius: 2 };

  // Sparkline data
  const riskData = Array.from({ length: 20 }, (_, i) =>
    mode === "crisis" ? 40 + i * 1.8 + Math.sin(i) * 5 : 10 + Math.sin(i * 0.8) * 8
  );

  // ── Fetch live data from API ───────────────────────────────────────────────
  async function loadLiveData() {
    try {
      const [latest, risk, week, today, baseline] = await Promise.all([
        fetch(`${API}/api/logs/latest`).then((r) => r.json()),
        fetch(`${API}/api/risk`).then((r) => r.json()),
        fetch(`${API}/api/logs/week`).then((r) => r.json()),
        fetch(`${API}/api/logs/today`).then((r) => r.json()),
        fetch(`${API}/api/baseline`).then((r) => r.json()),
      ]);

      const timeline   = logsToTimeline(Array.isArray(today) ? today : []);
      const ringFormat = timelineToRingFormat(timeline);
      const weekData   = buildWeekData(Array.isArray(week) ? week : [], risk);
      const medication = getMedSchedule(Array.isArray(today) ? today : [], mode);

      // Map anomalies from detector into alert feed format
      const alertsFromAPI = Array.isArray(risk?.anomalies)
        ? risk.anomalies
            .filter((a) => a && typeof a === "object")
            .map((a) => ({
              time:     "+0s",
              label:    (a.message || "Alert").slice(0, 40),
              sub:      a.message || "",
              color:    a.severity === "HIGH" ? "#ff1744" : "#ffeb3b",
              severity: a.severity === "HIGH" ? "critical" : "warning",
            }))
        : [];

      setLiveData({
        current:     latest?.activity     || "unknown",
        confidence:  latest?.confidence   ?? 0,
        risk:        risk?.risk_score     ?? 0,
        baselineRisk:baseline?.baseline_risk ?? 15,
        apiOk:       true,
        alerts:      alertsFromAPI.length > 0 ? alertsFromAPI : DEMO_NORMAL.alerts,
        week:        weekData.map((d) => ({
          day:  d.day.slice(0, 3).toUpperCase(),
          risk: d.risk,
          pill: d.pill,
        })),
        medication,
        illnesses:   DEMO_NORMAL.illnesses, // Static setup for hackathon, usually fetched from API
        vitals: DEMO_NORMAL.vitals,
      });
    } catch (_e) {
      setLiveData((prev) => ({ ...prev, apiOk: false }));
    }
  }

  // ── Demo data injection ────────────────────────────────────────────────────
  async function handleInjectDemo() {
    setInjecting(true);
    try {
      await fetch(`${API}/api/demo/inject`, { method: "POST" });
      await loadLiveData();
    } catch (_e) {
      // silently fail
    } finally {
      setInjecting(false);
    }
  }

  // ── Scan Label Integration ───────────────────────────────────────────────
  async function handleScanLabel(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsScanning(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API}/api/medication/scan`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        if (data.medication_name) setFormName(data.medication_name);
        if (data.dose) setFormDose(data.dose);
      } else {
        console.error("Failed to scan label", await res.text());
        alert("Failed to scan label. Please check API logs.");
      }
    } catch (err) {
      console.error("Scan error:", err);
      alert("Error scanning label. Is the backend running?");
    } finally {
      setIsScanning(false);
    }
    
    // Reset file input so user can scan same file again if needed
    e.target.value = "";
  }




  // ── Auto-fetch on mount and every 30s ─────────────────────────────────────
  useEffect(() => {
    loadLiveData();
    const interval = setInterval(loadLiveData, 30_000);
    return () => clearInterval(interval);
  }, [mode]);

  // ── Manual Override ────────────────────────────────────────────────────────
  function toggleMedication(index) {
    setLiveData((prev) => {
      const newMedication = [...prev.medication];
      const med = { ...newMedication[index] };
      med.done = !med.done;
      if (med.done) {
        const now = new Date();
        med.actual = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
      } else {
        med.actual = null;
      }
      newMedication[index] = med;
      return { ...prev, medication: newMedication };
    });
  }

  function removeMedicine(id) {
    setMyMedicines(prev => prev.filter(med => med.id !== id));
  }

  // ── Staggered alert reveal animation ──────────────────────────────────────
  useEffect(() => {
    setRevealed([]);
    data.alerts.forEach((_, i) => {
      setTimeout(() => setRevealed((r) => [...r, i]), i * 300 + 100);
    });
  }, [mode, liveData.apiOk]);



  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ background: "#080b12", minHeight: "100vh", fontFamily: mono, color: "#c9d1d9", display: "flex", flexDirection: "column", fontSize: 11 }}>

      {/* ── TOPBAR ── */}
      <div style={{ borderBottom: "1px solid #1e2535", padding: "0 16px", height: 44, display: "flex", alignItems: "center", justifyContent: "space-between", background: "#080b12" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 4, background: "linear-gradient(135deg,#ffeb3b,#ff9800)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>💊</div>
          <div>
            <span style={{ fontWeight: 700, fontSize: 13, color: "#fff", letterSpacing: 1 }}>CARE</span>
            <span style={{ fontWeight: 700, fontSize: 13, color: "#ffeb3b", letterSpacing: 1 }}>MEDS</span>
            <span style={{ fontSize: 8, color: "#2d3550", marginLeft: 8 }}>V1.2</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: 32, alignItems: "center" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 8, color: "#4a5568", letterSpacing: 2 }}>RESIDENT</div>
            <div style={{ fontSize: 10, color: "#8892a4" }}>MRS TAN · 74F · LIVING ROOM</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 8, color: "#4a5568", letterSpacing: 2 }}>PROFILE</div>
            <div style={{ fontSize: 10, color: "#8892a4" }}>DIET & MEDICATION FOCUS</div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {["normal", "crisis"].map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setAlertAck(false); }}
              style={{
                background:   mode === m ? (m === "crisis" ? "#ff1744" : "#00e676") : "transparent",
                color:        mode === m ? "#000" : "#4a5568",
                border:       `1px solid ${m === "crisis" ? "#ff174440" : "#00e67640"}`,
                borderRadius: 2, padding: "3px 8px", fontSize: 8,
                letterSpacing: 2, cursor: "pointer", fontFamily: mono,
              }}
            >
              {m === "normal" ? "NORMAL DAY" : "MISSED DOSE"}
            </button>
          ))}

          {/* Live indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: 4, padding: "3px 10px", background: "#0d1117", borderRadius: 10, border: "1px solid #1e2535" }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: liveData.apiOk ? "#00e676" : "#546e7a", animation: liveData.apiOk ? "pulse 2s infinite" : "none" }} />
            <span style={{ fontSize: 8, color: liveData.apiOk ? "#00e676" : "#546e7a", letterSpacing: 2 }}>
              {liveData.apiOk ? "LIVE" : "DEMO"}
            </span>
          </div>

          {/* Clock */}
          <Clock />
        </div>
      </div>

      {/* ── CRISIS BANNER ── */}
      {mode === "crisis" && !alertAck && (
        <div style={{ background: "linear-gradient(135deg,#1a0a0a,#2d0f0f)", borderBottom: "1px solid #ff1744", padding: "8px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 14 }}>🚨</span>
            <div>
              <span style={{ color: "#ff6b6b", fontWeight: 700, fontSize: 10, letterSpacing: 2 }}>CRITICAL ALERT — MRS TAN</span>
              <span style={{ color: "#ff9999", fontSize: 9, marginLeft: 12 }}>Pill taking not detected · Expected 08:45 · Now 2h 15m overdue</span>
            </div>
          </div>
          <button
            onClick={() => setAlertAck(true)}
            style={{ background: "transparent", border: "1px solid #ff174460", color: "#ff6b6b", padding: "3px 10px", fontSize: 8, letterSpacing: 2, cursor: "pointer", fontFamily: mono, borderRadius: 2 }}
          >
            ACKNOWLEDGE
          </button>
        </div>
      )}

      {/* ── Demo data banner ── */}
      {!liveData.apiOk && (
        <div style={{ background: "#0d1117", borderBottom: "1px solid #1e2535", padding: "6px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 8, color: "#4a5568" }}>No live data — API unreachable. Showing demo fallback.</span>
          <button
            onClick={handleInjectDemo}
            disabled={injecting}
            style={{ background: "transparent", border: "1px solid #4a9eff40", color: "#4a9eff", padding: "2px 8px", fontSize: 8, letterSpacing: 2, cursor: injecting ? "not-allowed" : "pointer", fontFamily: mono, borderRadius: 2 }}
          >
            {injecting ? "LOADING..." : "LOAD DEMO DATA"}
          </button>
        </div>
      )}

      {/* ── 3-PANEL LAYOUT ── */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "300px 1fr 280px", overflow: "hidden" }}>

        {/* ── LEFT PANEL: DIET & HEALTH PROFILE ── */}
        <div style={{ borderRight: "1px solid #1e2535", padding: 16, display: "flex", flexDirection: "column", gap: 16, overflowY: "auto", background: "#080b12" }}>
          <div style={{ fontSize: 10, color: "#4a9eff", letterSpacing: 3, borderBottom: "1px solid #1e2535", paddingBottom: 8 }}>HEALTH PROFILE & LIFESTYLE</div>
          
          <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 2 }}>DIAGNOSED ILLNESSES</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {data.illnesses.map(ill => (
              <span key={ill} style={{ background: "#4a9eff18", color: "#4a9eff", border: "1px solid #4a9eff40", padding: "4px 8px", borderRadius: 12, fontSize: 10 }}>
                {ill}
              </span>
            ))}
          </div>

          <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 2, marginTop: 8 }}>DIET & LIFESTYLE SUGGESTIONS</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {data.illnesses.map(ill => {
              const hasFood = FOOD_RECOMMENDATIONS[ill];
              const hasLife = LIFESTYLE_RECOMMENDATIONS[ill];
              if (!hasFood && !hasLife) return null;

              return (
                <div key={ill} style={{ ...panel, padding: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#00e676" }} />
                    <span style={{ fontSize: 11, color: "#00e676", fontWeight: 700 }}>{ill}</span>
                  </div>
                  
                  {hasFood && (
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 9, color: "#ffeb3b", letterSpacing: 1, marginBottom: 4 }}>🍽️ DIET</div>
                      <ul style={{ paddingLeft: 16, margin: 0, color: "#c9d1d9", fontSize: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                        {FOOD_RECOMMENDATIONS[ill].map((rec, i) => (
                          <li key={`food-${i}`}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {hasLife && (
                    <div>
                      <div style={{ fontSize: 9, color: "#4a9eff", letterSpacing: 1, marginBottom: 4 }}>🏃 LIFESTYLE & ACTIVITY</div>
                      <ul style={{ paddingLeft: 16, margin: 0, color: "#c9d1d9", fontSize: 10, display: "flex", flexDirection: "column", gap: 6 }}>
                        {LIFESTYLE_RECOMMENDATIONS[ill].map((rec, i) => (
                          <li key={`life-${i}`}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 3, marginTop: 12 }}>ENVIRONMENT</div>
          <div style={{ ...panel, padding: "8px 10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", padding: "3px 0", borderBottom: "1px solid #1e2535" }}>
              <span style={{ fontSize: 8, color: "#4a5568", letterSpacing: 1 }}>IN ROOM</span>
              <span style={{ fontSize: 12, color: "#c9d1d9", fontWeight: 600 }}>
                {data.vitals.inRoom.a}
              </span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", padding: "3px 0" }}>
              <span style={{ fontSize: 8, color: "#4a5568", letterSpacing: 1 }}>TEMP</span>
              <span style={{ fontSize: 12, color: "#c9d1d9", fontWeight: 600 }}>
                {data.vitals.temp.a}<span style={{ fontSize: 8, color: "#2d3550", marginLeft: 2 }}>{data.vitals.temp.unit}</span>
              </span>
            </div>
          </div>
        </div>

        {/* ── CENTRE PANEL: MEDICATION SCHEDULE ── */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "24px 0", gap: 0, overflowY: "auto", position: "relative" }}>
          <div style={{ width: "80%", display: "flex", gap: 32, borderBottom: "1px solid #1e2535", marginBottom: 24 }}>
            {["schedule", "add", "list"].map((tab) => {
              const labels = { schedule: "📅 Today's Schedule", add: "➕ Add Medicine", list: `💊 My Medicines (${myMedicines.length})` };
              const isActive = medTab === tab;
              return (
                <div 
                  key={tab} 
                  onClick={() => setMedTab(tab)}
                  style={{ 
                    paddingBottom: 8, 
                    cursor: "pointer",
                    color: isActive ? "#4a9eff" : "#8892a4",
                    borderBottom: isActive ? "2px solid #4a9eff" : "2px solid transparent",
                    fontWeight: isActive ? 700 : 400,
                    fontSize: 12, transition: "all 0.2s"
                  }}
                >
                  {labels[tab]}
                </div>
              );
            })}
          </div>

          {/* Today's Schedule View */}
          {medTab === "schedule" && (
            <div style={{ width: "80%", display: "flex", flexDirection: "column", gap: 12 }}>
              {data.medication.map((med, i) => {
              const isMissed = !med.done && mode === "crisis";
              const isTaken = med.done;
              const isUpcoming = !med.done && !isMissed;
              
              let statusColor = "#8892a4";
              if (isMissed) statusColor = "#ff1744";
              if (isTaken) statusColor = "#00e676";
              if (isUpcoming) statusColor = "#ffeb3b";

              return (
                <div 
                  key={i} 
                  onClick={() => toggleMedication(i)}
                  style={{ 
                    display: "flex", alignItems: "center", gap: 16, padding: "12px", 
                    background: `${statusColor}10`, border: `1px solid ${statusColor}40`, 
                    borderRadius: 6, cursor: "pointer", transition: "all 0.2s" 
                  }}
                  onMouseOver={(e) => { e.currentTarget.style.background = `${statusColor}20`; }}
                  onMouseOut={(e) => { e.currentTarget.style.background = `${statusColor}10`; }}
                >
                  <div style={{ fontSize: 24, flexShrink: 0 }}>
                    {isMissed ? "❌" : isTaken ? "✅" : "⏳"}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: statusColor }}>{med.label.toUpperCase()}</span>
                      <span style={{ fontSize: 12, color: "#c9d1d9", fontWeight: 700 }}>{med.time}</span>
                    </div>
                    <div style={{ fontSize: 10, color: "#8892a4", marginTop: 4, display: "flex", justifyContent: "space-between" }}>
                      <span>{med.actual ? `Taken at ${med.actual}` : `Scheduled for ${med.time}`}</span>
                      <span style={{ fontSize: 8, color: "#4a5568", fontStyle: "italic", textDecoration: "underline" }}>Click to change status</span>
                    </div>
                    <div style={{ fontSize: 9, color: "#4a9eff", marginTop: 4, letterSpacing: 1 }}>
                      REQUIRED FOR: {med.requiredFor.toUpperCase()}
                    </div>
                  </div>
                </div>
              );
            })}
            </div>
          )}

          {/* My Medicines List View */}
          {medTab === "list" && (
            <div style={{ width: "80%", display: "flex", flexDirection: "column", gap: 12 }}>
              {myMedicines.map(med => (
                <div key={med.id} style={{ display: "flex", background: "#0d1117", border: "1px solid #1e2535", borderRadius: 6, overflow: "hidden" }}>
                  <div style={{ width: 4, background: med.color }} />
                  <div style={{ flex: 1, padding: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "#c9d1d9", marginBottom: 6 }}>{med.name}</div>
                      <div style={{ fontSize: 10, color: "#8892a4" }}>{med.dosage} · {med.frequency} · {med.duration}</div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                      <div style={{ display: "flex", gap: 8, fontSize: 16 }}>
                        {med.times.includes("morning") && <span title="Morning">🌅</span>}
                        {med.times.includes("afternoon") && <span title="Afternoon">☀️</span>}
                        {med.times.includes("night") && <span title="Night">🌙</span>}
                      </div>
                      <button 
                        onClick={() => removeMedicine(med.id)}
                        style={{ 
                          background: "#ff174415", border: "1px solid #ff174440", color: "#ff1744", 
                          width: 28, height: 28, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
                          cursor: "pointer", fontSize: 14, transition: "all 0.2s" 
                        }}
                        title="Remove Medicine"
                        onMouseOver={(e) => { e.currentTarget.style.background = "#ff174430"; }}
                        onMouseOut={(e) => { e.currentTarget.style.background = "#ff174415"; }}
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Add Medicine Form View */}
          {medTab === "add" && (
            <div style={{ width: "80%", background: "#0d1117", border: "1px solid #1e2535", borderRadius: 6, padding: "24px", position: "relative" }}>
              <div style={{ position: "absolute", top: 24, right: 24 }}>
                <input 
                  type="file" 
                  id="label-upload" 
                  accept="image/*" 
                  style={{ display: "none" }} 
                  onChange={handleScanLabel}
                />
                <label 
                  htmlFor="label-upload"
                  style={{ 
                    background: isScanning ? "#ff980020" : "#ff980015", 
                    border: `1px solid ${isScanning ? "#ff980080" : "#ff980040"}`, 
                    color: "#ff9800", 
                    padding: "8px 16px", 
                    borderRadius: 4, 
                    cursor: isScanning ? "wait" : "pointer", 
                    fontWeight: 700, 
                    fontSize: 10, 
                    letterSpacing: 1,
                    display: "flex",
                    alignItems: "center",
                    gap: 6
                  }}
                >
                  {isScanning ? "⏳ SCANNING..." : "📷 SCAN LABEL"}
                </label>
              </div>

              <div style={{ fontSize: 10, color: "#8892a4", marginBottom: 8, letterSpacing: 1 }}>MEDICINE NAME *</div>
              <input 
                type="text" 
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                placeholder="e.g. Metformin 500mg" 
                style={{ width: "100%", background: "#080b12", border: "1px solid #1e2535", color: "#c9d1d9", padding: "10px", borderRadius: 4, marginBottom: 16, outline: "none", fontSize: 12 }} 
              />
              
              <div style={{ fontSize: 10, color: "#8892a4", marginBottom: 8, letterSpacing: 1 }}>DOSAGE</div>
              <input 
                type="text" 
                value={formDose}
                onChange={(e) => setFormDose(e.target.value)}
                placeholder="e.g. 1 tablet, 2 capsules" 
                style={{ width: "100%", background: "#080b12", border: "1px solid #1e2535", color: "#c9d1d9", padding: "10px", borderRadius: 4, marginBottom: 24, outline: "none", fontSize: 12 }} 
              />
              
              <div style={{ display: "flex", gap: 12 }}>
                <button 
                  onClick={() => alert("Save functionality goes here!")}
                  style={{ background: "#4a9eff15", border: "1px solid #4a9eff40", color: "#4a9eff", padding: "8px 24px", borderRadius: 4, cursor: "pointer", fontWeight: 700, fontSize: 10, letterSpacing: 1 }}
                >
                  SAVE MEDICINE
                </button>
                <button 
                  onClick={() => { setFormName(""); setFormDose(""); }}
                  style={{ background: "transparent", border: "1px solid #1e2535", color: "#8892a4", padding: "8px 24px", borderRadius: 4, cursor: "pointer", fontSize: 10, letterSpacing: 1 }}
                >
                  CLEAR
                </button>
              </div>
            </div>
          )}

          {/* 7-day week */}
          <div style={{ width: "80%", marginTop: 32 }}>
            <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 3, marginBottom: 8 }}>7-DAY ADHERENCE HISTORY</div>
            <div style={{ display: "flex", gap: 10 }}>
              {data.week.map((d, i) => {
                const c = d.risk <= 30 ? "#00e676" : d.risk <= 60 ? "#ffeb3b" : "#ff1744";
                return (
                  <div key={i} style={{ flex: 1, textAlign: "center", background: "#0d1117", padding: "8px 0", borderRadius: 4, border: "1px solid #1e2535" }}>
                    <div style={{ fontSize: 9, color: "#8892a4", marginBottom: 6 }}>{d.day}</div>
                    <div style={{ fontSize: 16 }}>{d.pill ? "✅" : "❌"}</div>
                    <div style={{ fontSize: 8, color: c, marginTop: 6, fontWeight: 700 }}>RISK: {d.risk}</div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Status bar */}
          <div style={{ width: "80%", marginTop: "auto", paddingTop: 12, borderTop: "1px solid #1e2535", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#00e676" }} />
              <span style={{ fontSize: 8, color: "#00e676", letterSpacing: 2 }}>READY</span>
            </div>
            <span style={{ fontSize: 7, color: "#2d3550" }}>SESS_MRS_TAN_001 · CareMeds v1.2</span>
          </div>
        </div>

        {/* ── RIGHT PANEL: ALERTS & RISK ── */}
        <div style={{ borderLeft: "1px solid #1e2535", padding: 16, display: "flex", flexDirection: "column", overflowY: "auto", background: "#080b12" }}>
          
          <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 16 }}>
            <RiskScore score={data.risk} label="CURRENT ADHERENCE RISK" />
            
            <div style={{ ...panel, padding: "8px 10px" }}>
              <div style={{ fontSize: 8, color: "#4a5568", letterSpacing: 2, marginBottom: 4 }}>RISK TREND</div>
              <Spark data={riskData} color={riskColor} />
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, marginTop: 16, borderBottom: "1px solid #1e2535", paddingBottom: 8 }}>
            <span style={{ fontSize: 10, color: "#ffeb3b", letterSpacing: 3 }}>EVENT FEED</span>
          </div>

          {/* Alert feed */}
          <div style={{ flex: 1 }}>
            {data.alerts.map((a, i) => (
              <AlertRow key={`${mode}-${i}`} alert={a} revealed={revealed.includes(i)} />
            ))}
          </div>

          {/* Summary stats */}
          <div style={{ marginTop: 16, borderTop: "1px solid #1e2535", paddingTop: 12 }}>
            <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 3, marginBottom: 8 }}>SUMMARY STATS</div>
            {[
              { label: "Pill compliance (7d)", value: `${Math.round((data.week.filter((d) => d.pill).length / Math.max(data.week.length, 1)) * 100)}%`, color: "#00e676" },
              { label: "Missed Doses (7d)",    value: `${data.week.filter((d) => !d.pill).length}`, color: "#ff1744" },
              { label: "Avg Risk Level",       value: `${Math.round(data.week.reduce((s, d) => s + d.risk, 0) / Math.max(data.week.length, 1))}`,         color: "#ffeb3b" },
            ].map((s, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: i < 2 ? "1px solid #1e2535" : "none" }}>
                <span style={{ fontSize: 9, color: "#4a5568" }}>{s.label}</span>
                <span style={{ fontSize: 11, color: s.color, fontWeight: 700 }}>{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
