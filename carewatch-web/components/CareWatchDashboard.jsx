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
    { label: "Morning dose", time: "08:00", done: true,  actual: "08:05", requiredFor: "Hypertension (High BP)", status: "taken" },
    { label: "Lunch dose",   time: "13:00", done: true,  actual: "13:12", requiredFor: "Diabetes",                status: "taken" },
    { label: "Night dose",   time: "21:00", done: false, actual: null,    requiredFor: "High Cholesterol",       status: "missed" },
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
    { time: "+2.0s", label: "NIGHT DOSE MISSED",    sub: "Night dose at 21:00 was not detected.", color: "#ff1744", severity: "critical" },
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
 * Build medication schedule dynamically from API schedules and today's logs.
 */
function getMedSchedule(todayLogs, schedules) {
  let mappedSchedules = [];
  if (!Array.isArray(schedules) || schedules.length === 0) {
    mappedSchedules = [
      { id: 1, time_of_day: "08:00", medication_name: "Morning dose", illness_hint: "General" },
      { id: 2, time_of_day: "13:00", medication_name: "Lunch dose",   illness_hint: "General" },
      { id: 3, time_of_day: "21:00", medication_name: "Night dose",   illness_hint: "General" },
    ];
  } else {
    mappedSchedules = schedules;
  }
  
  const pillLogs = (todayLogs || []).filter((r) => r.activity === "pill_taking");
  const now = new Date();
  const nowMin = now.getHours() * 60 + now.getMinutes();
  
  const result = mappedSchedules.map(s => {
    let done = false;
    let actual = null;
    
    const schedHour = parseInt(s.time_of_day.split(":")[0]);
    const schedMin = parseInt(s.time_of_day.split(":")[1] || 0);
    const toleranceMin = Number.isFinite(s.tolerance_min) ? s.tolerance_min : 30;
    const schedTotalMin = schedHour * 60 + schedMin;

    for (const p of pillLogs) {
      const h = p.hour ?? 0;
      const m = p.minute ?? 0;
      const diff = Math.abs((h * 60 + m) - (schedHour * 60 + schedMin));
      if (diff <= 120 && !done) {
        done = true;
        actual = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
      }
    }

    const missed = !done && nowMin > (schedTotalMin + toleranceMin);
    const status = done ? "taken" : missed ? "missed" : "upcoming";
    
    return {
      id: s.id,
      medication_name: s.medication_name,
      time: s.time_of_day,
      label: s.medication_name,
      done,
      actual,
      requiredFor: s.illness_hint || "General",
      tolerance_min: toleranceMin,
      status,
    };
  });
  return result;
}

// ── Sub-components ───────────────────────────────────────────────────────────

function Spark({ data, color, height = 36 }) {
  const w = 160, h = height;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => {
      const x = ((i / (data.length - 1)) * w).toFixed(2);
      const y = (h - ((v - min) / range) * (h - 4) - 2).toFixed(2);
      return `${x},${y}`;
    })
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
  const [medTab,      setMedTab]      = useState("schedule");
  const [myMedicines, setMyMedicines] = useState(DEMO_MY_MEDICINES);
  const [revealed, setRevealed] = useState([]);
  const [liveData, setLiveData] = useState(INITIAL_LIVE);
  const [injecting,setInjecting]= useState(false);
  const [pendingDoseChange, setPendingDoseChange] = useState(null); // { index, nextDone }

  // Consent gate — states: null (mounting), "required", "declined", "granted"
  // Using useEffect to avoid hydration mismatch.
  const [consentStatus, setConsentStatus] = useState(null);

  useEffect(() => {
    const v = localStorage.getItem("carewatch_consent");
    if (v === "granted") setConsentStatus("granted");
    else if (v === "declined") setConsentStatus("declined");
    else setConsentStatus("required");

    // Load any local-only saved data (works when backend is not running)
    try {
      const lm = JSON.parse(localStorage.getItem("carewatch_local_medicines") || "null");
      if (Array.isArray(lm) && lm.length > 0) setMyMedicines(lm);
    } catch (_e) {}
    try {
      const lmeals = JSON.parse(localStorage.getItem("carewatch_local_meals") || "null");
      if (Array.isArray(lmeals) && lmeals.length > 0) setMeals(lmeals);
    } catch (_e) {}
  }, []);

  function handleConsent() {
    localStorage.setItem("carewatch_consent", "granted");
    setConsentStatus("granted");
  }

  function handleDecline() {
    localStorage.setItem("carewatch_consent", "declined");
    setConsentStatus("declined");
  }

  // Add Medicine Form State
  const [formName, setFormName] = useState("");
  const [formDose, setFormDose] = useState("");
  const [formTimes, setFormTimes] = useState([]);
  const [formIllness, setFormIllness] = useState("");
  // meal_relation for add-medicine form: "fixed" | "before" | "after"
  const [formMealRelation, setFormMealRelation] = useState("fixed");
  // Default meal times — user can override
  const [mealTimes, setMealTimes] = useState({
    Breakfast: "07:30",
    Lunch:     "12:30",
    Dinner:    "19:00",
  });

  // Illnesses Interactive State
  const [confirmedIllnesses, setConfirmedIllnesses] = useState([]);
  const [removedIllnesses, setRemovedIllnesses] = useState([]);

  // Meals State
  const [meals, setMeals] = useState([]);
  const [mealFormName, setMealFormName] = useState("Breakfast");
  const [mealFormTime, setMealFormTime] = useState("08:00");

  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Use live data when API is reachable; otherwise fall back to demo constants.
  // In demo mode, if we have a locally overridden medication list in
  // liveData.medication (from manual toggles), merge that into the demo snapshot
  // so the UI reflects the change.
  const _rawData = liveData.apiOk
    ? { ...liveData }
    : (Array.isArray(liveData.medication) && liveData.medication.length > 0)
        ? { ...DEMO_NORMAL, medication: liveData.medication }
        : DEMO_NORMAL;
    
  // Filter out removed illnesses for the entire UI
  const data = {
    ..._rawData,
    illnesses: (_rawData.illnesses || []).filter(ill => !removedIllnesses.includes(ill))
  };

  const riskColor = data.risk <= 30 ? "#00e676" : data.risk <= 60 ? "#ffeb3b" : "#ff1744";
  const mono      = "'IBM Plex Mono', 'Courier New', monospace";
  const panel     = { background: "#0d1117", border: "1px solid #1e2535", borderRadius: 2 };

  // Sparkline data
  const riskData = Array.from({ length: 20 }, (_, i) =>
    10 + Math.sin(i * 0.8) * 8
  );

  // ── Fetch live data from API ───────────────────────────────────────────────
  async function loadLiveData() {
    try {
      const [latest, risk, week, today, baseline, schedules, fetchedMeals, recommendations, todaySchedule] = await Promise.all([
        fetch(`${API}/api/logs/latest`).then((r) => r.json()),
        fetch(`${API}/api/risk`).then((r) => r.json()),
        fetch(`${API}/api/logs/week`).then((r) => r.json()),
        fetch(`${API}/api/logs/today`).then((r) => r.json()),
        fetch(`${API}/api/baseline`).then((r) => r.json()),
        fetch(`${API}/api/medication/schedules`).then((r) => r.json()).catch(() => []),
        fetch(`${API}/api/meals`).then((r) => r.json()).catch(() => []),
        fetch(`${API}/api/medication/recommendations`).then((r) => r.json()).catch(() => ({ illnesses: [] })),
        fetch(`${API}/api/medication/today`).then((r) => r.json()).catch(() => []),
      ]);

      setMeals(Array.isArray(fetchedMeals) ? fetchedMeals : []);

      const timeline   = logsToTimeline(Array.isArray(today) ? today : []);
      const ringFormat = timelineToRingFormat(timeline);
      const weekData   = buildWeekData(Array.isArray(week) ? week : [], risk);
      const safeSchedules = Array.isArray(schedules) ? schedules : [];

      // Build today's medication display from the dedicated /api/medication/today endpoint.
      // Each entry has: id, medication_name, dose, time_of_day, tolerance_min, illness_hint,
      //                 meal_relation, meal_name, status ('taken'|'missed'|'upcoming'), actual_time
      let medication;
      const safeTodaySchedule = Array.isArray(todaySchedule) ? todaySchedule : [];
      if (safeTodaySchedule.length > 0) {
        medication = safeTodaySchedule.map(s => ({
          id:           s.id,
          medication_name: s.medication_name,
          label:        s.medication_name,
          time:         s.time_of_day,
          actual:       s.actual_time || null,
          requiredFor:  s.illness_hint || "General",
          tolerance_min: s.tolerance_min ?? 30,
          meal_relation: s.meal_relation || "fixed",
          meal_name:    s.meal_name || null,
          status:       s.status,           // 'taken' | 'missed' | 'upcoming'
          done:         s.status === "taken",
        }));
      } else {
        // No live schedules — fall back to client-side computed schedule
        medication = getMedSchedule(Array.isArray(today) ? today : [], safeSchedules);
      }

      const illnessesFromApi = Array.isArray(recommendations?.illnesses) ? recommendations.illnesses : [];
      const illnessesFromSchedules = safeSchedules
        .map((s) => s?.illness_hint)
        .filter((ill) => typeof ill === "string" && ill.trim() && ill !== "General");
      const illnesses = [...new Set([
        ...illnessesFromApi,
        ...illnessesFromSchedules,
      ])];
      
      if (safeSchedules.length > 0) {
        const grouped = {};
        safeSchedules.forEach(s => {
          if (!grouped[s.medication_name]) {
            grouped[s.medication_name] = {
              id: s.id, // reference for delete
              name: s.medication_name,
              dosage: s.dose || "1 tablet",
              frequency: "",
              duration: "ongoing",
              times: [],
              color: "#4a9eff",
              scheduleIds: []
            };
          }
          grouped[s.medication_name].scheduleIds.push(s.id);
          const hour = parseInt(s.time_of_day.split(":")[0]);
          let timeName = "morning";
          if (hour >= 12 && hour < 17) timeName = "afternoon";
          else if (hour >= 17) timeName = "night";
          grouped[s.medication_name].times.push(timeName);
        });
        
        Object.values(grouped).forEach(g => {
          g.frequency = `${g.times.length} time(s)/day`;
        });
        setMyMedicines(Object.values(grouped));
      } else {
        setMyMedicines(DEMO_MY_MEDICINES);
      }

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

      // Add synthetic alerts for any taken / missed doses in today's schedule
      const takenDoseAlerts = (medication || [])
        .filter((m) => m.status === "taken")
        .map((m) => ({
          time:     m.actual || m.time || "--:--",
          label:    "DOSE TAKEN",
          sub:      `${m.label || m.medication_name} taken at ${m.actual || m.time}.`,
          color:    "#00e676",
          severity: "ok",
        }));

      const missedDoseAlerts = (medication || [])
        .filter((m) => m.status === "missed")
        .map((m) => ({
          time:     m.time || "--:--",
          label:    "MISSED DOSE",
          sub:      `${m.label || m.medication_name} was not detected within the safety window.`,
          color:    "#ff1744",
          severity: "critical",
        }));

      const combinedAlerts =
        (alertsFromAPI.length > 0 ? alertsFromAPI : DEMO_NORMAL.alerts) // keep existing behaviour
          .concat(takenDoseAlerts)
          .concat(missedDoseAlerts);

      setLiveData({
        current:     latest?.activity     || "unknown",
        confidence:  latest?.confidence   ?? 0,
        risk:        risk?.risk_score     ?? 0,
        baselineRisk:baseline?.baseline_risk ?? 15,
        apiOk:       true,
        alerts:      combinedAlerts,
        week:        weekData.map((d) => ({
          day:  d.day.slice(0, 3).toUpperCase(),
          risk: d.risk,
          pill: d.pill,
        })),
        medication,
        illnesses:   illnesses.length > 0 ? illnesses : DEMO_NORMAL.illnesses,
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

  // ── Auto-fetch on mount and every 30s ─────────────────────────────────────
  useEffect(() => {
    loadLiveData();
    const interval = setInterval(loadLiveData, 30_000);
    return () => clearInterval(interval);
  }, []);

  // ── Manual Override ────────────────────────────────────────────────────────
  function toggleMedication(index) {
    const med = data.medication[index];
    if (!med) return;
    setPendingDoseChange({ index, nextDone: !med.done });
  }

  async function applyMedicationChange(index, nextDone) {
    const med = data.medication[index];
    if (!med) return;

    const now = new Date();
    const actualTime = `${String(now.getHours()).padStart(2, "0")}:${String(
      now.getMinutes()
    ).padStart(2, "0")}`;

    // Optimistically update local schedule so UI shows change immediately,
    // even in demo mode / when backend is down.
    setLiveData((prev) => {
      const baseList =
        Array.isArray(prev.medication) && prev.medication.length > 0
          ? prev.medication
          : data.medication;

      const nowMin = now.getHours() * 60 + now.getMinutes();
      const newMedication = baseList.map((m, i) =>
        i === index
          ? {
              ...m,
              done: nextDone,
              actual: nextDone ? actualTime : null,
              status: nextDone
                ? "taken"
                : (() => {
                    const t = (m.time || "00:00").split(":").map(Number);
                    const schedMin = (t[0] || 0) * 60 + (t[1] || 0);
                    const tol = Number.isFinite(m.tolerance_min) ? m.tolerance_min : 30;
                    return nowMin > (schedMin + tol) ? "missed" : "upcoming";
                  })(),
            }
          : m
      );
      return { ...prev, medication: newMedication };
    });

    // If API is live, also tell the backend so risk scores etc. stay in sync.
    if (liveData.apiOk) {
      try {
        if (nextDone) {
          // Persist manual override by schedule id so it survives the next poll.
          await fetch(`${API}/api/medication/event/${med.id}`, { method: "POST" });
        } else if (med.id != null) {
          // Undo today's event for this scheduled dose so refresh doesn't revert the UI.
          await fetch(`${API}/api/medication/event/${med.id}`, { method: "DELETE" });
        }
        setTimeout(() => loadLiveData(), 800);
      } catch (err) {
        console.error("Failed to toggle medication event", err);
      }
    }
  }

  async function removeMedicine(id) {
    try {
      const med = myMedicines.find(m => m.id === id);
      if (med && med.scheduleIds) {
        for (const sId of med.scheduleIds) {
          await fetch(`${API}/api/medication/schedules/${sId}`, { method: "DELETE" });
        }
      } else {
        await fetch(`${API}/api/medication/schedules/${id}`, { method: "DELETE" });
      }
      await loadLiveData();
    } catch (err) {
      console.error("Failed to remove medicine", err);
    }
    setMyMedicines(prev => prev.filter(med => med.id !== id));
    try {
      const next = myMedicines.filter(m => m.id !== id);
      localStorage.setItem("carewatch_local_medicines", JSON.stringify(next));
    } catch (_e) {
      // ignore localStorage failures
    }
  }

  async function handleSaveMedicine() {
    if (!formName) return alert("Please enter medicine name");
    if (formTimes.length === 0) return alert("Please select at least one time");
    
    // Derive (meal_name, time_of_day) pairs from formTimes + formMealRelation
    const MEAL_MAP = {
      morning:   { meal: "Breakfast", fixedTime: "08:00" },
      afternoon: { meal: "Lunch",     fixedTime: "13:00" },
      night:     { meal: "Dinner",    fixedTime: "20:00" },
    };

    // Always update local UI immediately (works even if backend is down)
    const newLocalMed = {
      id: Date.now(),
      name: formName,
      dosage: formDose || "1 tablet",
      frequency: `${formTimes.length} time(s)/day`,
      duration: "ongoing",
      times: [...formTimes],
      color: "#4a9eff",
    };
    setMyMedicines((prev) => {
      const next = [newLocalMed, ...prev];
      try {
        localStorage.setItem("carewatch_local_medicines", JSON.stringify(next));
      } catch (_e) {
        // ignore localStorage failures
      }
      return next;
    });

    try {
      for (const timeMode of formTimes) {
        const { meal, fixedTime } = MEAL_MAP[timeMode] || { meal: null, fixedTime: "08:00" };
        let time_of_day = fixedTime;
        let meal_name   = null;

        if (formMealRelation !== "fixed" && meal) {
          meal_name = meal;
          const baseMealTime = mealTimes[meal] || fixedTime;
          const [mh, mm] = baseMealTime.split(":").map(Number);
          let totalMin  = mh * 60 + mm;
          if (formMealRelation === "before") totalMin -= 15;  // remind 15 min before meal
          totalMin = Math.max(0, totalMin);
          const h = String(Math.floor(totalMin / 60)).padStart(2, "0");
          const m = String(totalMin % 60).padStart(2, "0");
          time_of_day = `${h}:${m}`;
        }

        await fetch(`${API}/api/medication/schedules`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            medication_name: formName,
            dose: formDose,
            time_of_day,
            illness_hint: formIllness || "General",
            meal_relation: formMealRelation,
            meal_name,
          })
        });
      }
      setFormName("");
      setFormDose("");
      setFormIllness("");
      setFormTimes([]);
      setFormMealRelation("fixed");
      setMedTab("list");
      await loadLiveData();
    } catch (err) {
      console.error("Failed to save schedule", err);
      // Frontend-only mode: backend might be down. Keep local UI update and avoid blocking.
      setMedTab("list");
    }
  }

  // ── Meal Handlers ─────────────────────────────────────────────────────────
  async function handleSaveMeal() {
    if (!mealFormName || !mealFormTime) return alert("Please fill in meal details");
    try {
      await fetch(`${API}/api/meals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          meal_name: mealFormName,
          time_of_day: mealFormTime,
          tolerance_min: 60
        })
      });
      await loadLiveData();
    } catch (err) {
      console.error("Failed to save meal", err);
      // Frontend-only mode fallback: store locally so UI works without backend.
      const localMeal = {
        id: Date.now(),
        meal_name: mealFormName,
        time_of_day: mealFormTime,
        tolerance_min: 60,
        localOnly: true,
      };
      setMeals((prev) => {
        const next = [localMeal, ...prev];
        try {
          localStorage.setItem("carewatch_local_meals", JSON.stringify(next));
        } catch (_e) {
          // ignore
        }
        return next;
      });
      setMealTimes((prev) => ({ ...prev, [mealFormName]: mealFormTime }));
    }
  }

  async function handleRemoveMeal(id) {
    try {
      await fetch(`${API}/api/meals/${id}`, { method: "DELETE" });
      await loadLiveData();
    } catch (err) {
      console.error("Failed to remove meal", err);
    }
    // Also remove from local list (for frontend-only mode)
    setMeals((prev) => {
      const next = prev.filter((m) => m.id !== id);
      try {
        localStorage.setItem("carewatch_local_meals", JSON.stringify(next));
      } catch (_e) {
        // ignore
      }
      return next;
    });
  }

  // ── Staggered alert reveal animation ──────────────────────────────────────
  useEffect(() => {
    setRevealed([]);
    data.alerts.forEach((_, i) => {
      setTimeout(() => setRevealed((r) => [...r, i]), i * 300 + 100);
    });
  }, [liveData.apiOk]);



  // ── Render ─────────────────────────────────────────────────────────────────

  // ── Consent gate ── hard block until consent is granted ───────────────────
  // null      = still mounting (show nothing to avoid any flash)
  // "required"= consent required → show modal with Accept/Decline
  // "declined"= blocked → show locked screen until Accept
  // "granted" = show dashboard
  if (consentStatus === null) return null;
  if (consentStatus === "required") {
    return (
      <div style={{
        background: "#080b12", minHeight: "100vh", display: "flex",
        alignItems: "center", justifyContent: "center",
        fontFamily: "'IBM Plex Mono', 'Courier New', monospace", color: "#c9d1d9",
        padding: 24,
      }}>
        <div style={{
          background: "#0d1117", border: "1px solid #1e2535", borderRadius: 6,
          padding: "40px 48px", maxWidth: 540, width: "100%",
        }}>
          {/* Logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 28 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 6,
              background: "linear-gradient(135deg,#ffeb3b,#ff9800)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
            }}>💊</div>
            <div>
              <span style={{ fontWeight: 700, fontSize: 16, color: "#fff", letterSpacing: 1 }}>CARE</span>
              <span style={{ fontWeight: 700, fontSize: 16, color: "#ffeb3b", letterSpacing: 1 }}>MEDS</span>
            </div>
          </div>

          <div style={{ fontSize: 13, fontWeight: 700, color: "#fff", letterSpacing: 2, marginBottom: 8 }}>
            DATA COLLECTION CONSENT
          </div>
          <div style={{ fontSize: 9, color: "#4a5568", letterSpacing: 2, marginBottom: 20 }}>
            REQUIRED BEFORE SYSTEM ACTIVATION
          </div>

          <div style={{ fontSize: 10, color: "#8892a4", lineHeight: 1.8, marginBottom: 20 }}>
            CareWatch monitors the resident using a webcam and AI to track:
          </div>
          <ul style={{ fontSize: 10, color: "#8892a4", lineHeight: 1.9, marginBottom: 24, paddingLeft: 20 }}>
            <li>Body keypoints and activity classification (sitting, eating, pill-taking, etc.)</li>
            <li>Medication intake and schedule adherence</li>
            <li>Mealtime patterns and activity risk scores</li>
          </ul>
          <div style={{ fontSize: 10, color: "#8892a4", lineHeight: 1.8, marginBottom: 24 }}>
            <strong style={{ color: "#c9d1d9" }}>No raw video is stored.</strong> Frames are processed in memory and discarded immediately.
            All data is stored locally using a pseudonymous ID and automatically deleted after 30 days.
            You may withdraw consent at any time by clearing your browser data.
          </div>

          <div style={{
            background: "#0a0f1a", border: "1px solid #1e2535",
            borderRadius: 4, padding: "12px 16px", marginBottom: 24,
          }}>
            <div style={{ fontSize: 9, color: "#ffeb3b", letterSpacing: 2, marginBottom: 6 }}>⚖ PDPA ALIGNED</div>
            <div style={{ fontSize: 9, color: "#4a5568", lineHeight: 1.7 }}>
              Data is used solely for health monitoring. Not shared with third parties.
              Collected under informed consent per the Singapore Personal Data Protection Act.
            </div>
          </div>

          <button
            id="consent-agree-btn"
            onClick={handleConsent}
            style={{
              width: "100%", padding: "12px 0",
              background: "linear-gradient(135deg,#00e676,#00bcd4)",
              color: "#000", border: "none", borderRadius: 4,
              fontFamily: "'IBM Plex Mono','Courier New',monospace",
              fontWeight: 700, fontSize: 11, letterSpacing: 2, cursor: "pointer",
            }}
          >
            I AGREE — ACTIVATE CAREWATCH
          </button>

          <button
            id="consent-decline-btn"
            onClick={handleDecline}
            style={{
              width: "100%",
              padding: "10px 0",
              marginTop: 10,
              background: "transparent",
              color: "#ff6b6b",
              border: "1px solid #ff174460",
              borderRadius: 4,
              fontFamily: "'IBM Plex Mono','Courier New',monospace",
              fontWeight: 700,
              fontSize: 10,
              letterSpacing: 2,
              cursor: "pointer",
            }}
          >
            I DO NOT AGREE — EXIT
          </button>

          <div style={{ fontSize: 8, color: "#2d3550", textAlign: "center", marginTop: 12 }}>
            By clicking above, you confirm that the resident or their authorised caregiver consents to monitoring.
          </div>
        </div>
      </div>
    );
  }

  if (consentStatus === "declined") {
    return (
      <div style={{
        background: "#080b12", minHeight: "100vh", display: "flex",
        alignItems: "center", justifyContent: "center",
        fontFamily: "'IBM Plex Mono', 'Courier New', monospace", color: "#c9d1d9",
        padding: 24,
      }}>
        <div style={{
          background: "#0d1117", border: "1px solid #1e2535", borderRadius: 6,
          padding: "40px 48px", maxWidth: 540, width: "100%",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 6,
              background: "linear-gradient(135deg,#ff1744,#ff9800)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
            }}>🔒</div>
            <div>
              <span style={{ fontWeight: 700, fontSize: 16, color: "#fff", letterSpacing: 1 }}>CARE</span>
              <span style={{ fontWeight: 700, fontSize: 16, color: "#ffeb3b", letterSpacing: 1 }}>MEDS</span>
            </div>
          </div>

          <div style={{ fontSize: 13, fontWeight: 700, color: "#fff", letterSpacing: 2, marginBottom: 10 }}>
            ACCESS DISABLED
          </div>
          <div style={{ fontSize: 10, color: "#8892a4", lineHeight: 1.8, marginBottom: 18 }}>
            You declined data collection consent. For safety and compliance, CareMeds features are locked until consent is granted.
          </div>

          <button
            id="consent-accept-from-declined-btn"
            onClick={handleConsent}
            style={{
              width: "100%", padding: "12px 0",
              background: "linear-gradient(135deg,#00e676,#00bcd4)",
              color: "#000", border: "none", borderRadius: 4,
              fontFamily: "'IBM Plex Mono','Courier New',monospace",
              fontWeight: 700, fontSize: 11, letterSpacing: 2, cursor: "pointer",
            }}
          >
            ACCEPT CONSENT — UNLOCK
          </button>

          <div style={{ fontSize: 8, color: "#2d3550", textAlign: "center", marginTop: 12 }}>
            If you are not the authorised caregiver, please close this page.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ background: "#080b12", minHeight: "100vh", fontFamily: mono, color: "#c9d1d9", display: "flex", flexDirection: "column", fontSize: 11 }}>

      {/* ── Confirm status change modal ── */}
      {pendingDoseChange && (() => {
        const med = data.medication[pendingDoseChange.index];
        if (!med) return null;
        const nextDone = pendingDoseChange.nextDone;
        const name = med.medication_name || med.label || "this dose";
        return (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.65)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 9999,
              padding: 24,
            }}
            onClick={() => setPendingDoseChange(null)}
          >
            <div
              style={{
                width: "100%",
                maxWidth: 520,
                background: "#0d1117",
                border: "1px solid #1e2535",
                borderRadius: 6,
                padding: "22px 22px",
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{ fontSize: 10, color: "#2d3550", letterSpacing: 2, marginBottom: 8 }}>CONFIRM ACTION</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 10 }}>
                {nextDone ? "Mark dose as TAKEN?" : "Mark dose as NOT taken?"}
              </div>
              <div style={{ fontSize: 10, color: "#8892a4", lineHeight: 1.6, marginBottom: 16 }}>
                You are about to {nextDone ? "mark" : "unmark"} <strong style={{ color: "#c9d1d9" }}>{name}</strong>.
              </div>
              <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
                <button
                  onClick={() => setPendingDoseChange(null)}
                  style={{
                    background: "transparent",
                    border: "1px solid #1e2535",
                    color: "#8892a4",
                    padding: "8px 14px",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontFamily: mono,
                    fontSize: 10,
                    letterSpacing: 1,
                  }}
                >
                  CANCEL
                </button>
                <button
                  onClick={async () => {
                    const { index, nextDone } = pendingDoseChange;
                    setPendingDoseChange(null);
                    await applyMedicationChange(index, nextDone);
                  }}
                  style={{
                    background: nextDone ? "linear-gradient(135deg,#00e676,#00bcd4)" : "linear-gradient(135deg,#ff1744,#ff9800)",
                    border: "none",
                    color: "#000",
                    padding: "8px 14px",
                    borderRadius: 4,
                    cursor: "pointer",
                    fontFamily: mono,
                    fontSize: 10,
                    letterSpacing: 1,
                    fontWeight: 700,
                  }}
                >
                  CONFIRM
                </button>
              </div>
            </div>
          </div>
        );
      })()}

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
            <div style={{ fontSize: 10, color: "#8892a4" }}>RESIDENT-001 · LIVING ROOM</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 8, color: "#4a5568", letterSpacing: 2 }}>PROFILE</div>
            <div style={{ fontSize: 10, color: "#8892a4" }}>DIET & MEDICATION FOCUS</div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
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

      {/* ── MISSED DOSE BANNER (if any) ── */}
      {data.medication.some((m) => m.status === "missed") && (() => {
        const missed = data.medication.filter((m) => m.status === "missed");
        const names = [...new Set(missed.map((m) => m.label || m.medication_name))];
        const summary =
          names.length === 1
            ? names[0]
            : names.length === 2
              ? `${names[0]} and ${names[1]}`
              : `${names[0]}, ${names[1]} +${names.length - 2} more`;
        return (
          <div style={{ background: "linear-gradient(135deg,#1a0a0a,#2d0f0f)", borderBottom: "1px solid #ff1744", padding: "8px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 14 }}>🚨</span>
              <div>
                <span style={{ color: "#ff6b6b", fontWeight: 700, fontSize: 10, letterSpacing: 2 }}>MISSED MEDICATION DETECTED</span>
                <span style={{ color: "#ff9999", fontSize: 9, marginLeft: 12 }}>
                  Missed: {summary}. Not detected within the safety window — please check on the resident.
                </span>
              </div>
            </div>
          </div>
        );
      })()}

      {/* ── 3-PANEL LAYOUT ── */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "300px 1fr 280px", overflow: "hidden" }}>

        {/* ── LEFT PANEL: DIET & HEALTH PROFILE ── */}
        <div style={{ borderRight: "1px solid #1e2535", padding: 16, display: "flex", flexDirection: "column", gap: 16, overflowY: "auto", background: "#080b12" }}>
          <div style={{ fontSize: 10, color: "#4a9eff", letterSpacing: 3, borderBottom: "1px solid #1e2535", paddingBottom: 8 }}>HEALTH PROFILE & LIFESTYLE</div>
          
          <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 2 }}>PREDICTED ILLNESSES</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {data.illnesses.map(ill => {
              const isConfirmed = confirmedIllnesses.includes(ill);
              return (
                <div key={ill} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: isConfirmed ? "#00e67615" : "#4a9eff18", border: `1px solid ${isConfirmed ? "#00e67640" : "#4a9eff40"}`, padding: "4px 8px", borderRadius: 4 }}>
                  <span style={{ color: isConfirmed ? "#00e676" : "#4a9eff", fontSize: 10, fontWeight: isConfirmed ? 700 : 400 }}>
                    {ill} {isConfirmed && "✓"}
                  </span>
                  {!isConfirmed && (
                    <div style={{ display: "flex", gap: 4 }}>
                      <button onClick={() => setConfirmedIllnesses(prev => [...prev, ill])} style={{ background: "transparent", border: "1px solid #00e67640", color: "#00e676", fontSize: 8, padding: "2px 6px", borderRadius: 2, cursor: "pointer", fontFamily: mono }}>CONFIRM</button>
                      <button onClick={() => setRemovedIllnesses(prev => [...prev, ill])} style={{ background: "transparent", border: "1px solid #ff174440", color: "#ff1744", fontSize: 8, padding: "2px 6px", borderRadius: 2, cursor: "pointer", fontFamily: mono }}>REMOVE</button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: 8, color: "#2d3550", letterSpacing: 2, borderBottom: "1px solid #1e2535", paddingBottom: 8, marginBottom: 8 }}>MEAL TIMES (TTS REMINDERS)</div>
            
            {/* Meal List */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
              {meals.map(m => (
                 <div key={m.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0d1117", border: "1px solid #1e2535", padding: "6px 8px", borderRadius: 4 }}>
                   <div>
                     <div style={{ color: "#c9d1d9", fontSize: 10, fontWeight: 700 }}>{m.meal_name}</div>
                     <div style={{ color: "#8892a4", fontSize: 9 }}>{m.time_of_day}</div>
                   </div>
                   <button onClick={() => handleRemoveMeal(m.id)} style={{ background: "transparent", border: "1px solid #ff174440", color: "#ff1744", fontSize: 8, padding: "2px 6px", borderRadius: 2, cursor: "pointer", fontFamily: mono }}>REMOVE</button>
                 </div>
              ))}
              {meals.length === 0 && <div style={{ fontSize: 9, color: "#546e7a", fontStyle: "italic" }}>No meal times configured.</div>}
            </div>
            
            {/* Add Meal Form */}
            <div style={{ background: "#0d1117", border: "1px dashed #1e2535", padding: 8, borderRadius: 4, display: "flex", gap: 6, flexDirection: "column" }}>
              <div style={{ display: "flex", gap: 6 }}>
                <select 
                  value={mealFormName} onChange={e => setMealFormName(e.target.value)}
                  style={{ flex: 1, background: "transparent", border: "1px solid #1e2535", color: "#c9d1d9", fontSize: 9, padding: 4, fontFamily: mono }}
                >
                  <option value="Breakfast">Breakfast</option>
                  <option value="Lunch">Lunch</option>
                  <option value="Dinner">Dinner</option>
                  <option value="Supper">Supper</option>
                  <option value="Snack">Snack</option>
                </select>
                <input 
                  type="time" value={mealFormTime} onChange={e => setMealFormTime(e.target.value)}
                  style={{ width: 60, background: "transparent", border: "1px solid #1e2535", color: "#c9d1d9", fontSize: 9, padding: 4, fontFamily: mono, WebkitAppearance: "none" }}
                  required
                />
              </div>
              <button 
                onClick={handleSaveMeal}
                style={{ background: "#4a9eff", border: "none", color: "#000", fontSize: 9, padding: "4px", borderRadius: 2, cursor: "pointer", fontFamily: mono, fontWeight: 700 }}
              >
                + ADD MEAL
              </button>
            </div>
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
              const isMissed = med.status === "missed";
              const isTaken = med.status === "taken";
              const isUpcoming = med.status === "upcoming";
              
              let statusColor = "#8892a4";
              if (isMissed) statusColor = "#ff1744";
              if (isTaken) statusColor = "#00e676";
              if (isUpcoming) statusColor = "#ffeb3b";

              return (
                <div
                  key={i}
                  onClick={() => toggleMedication(i)}
                  style={{
                    display: "flex", alignItems: "center", gap: 16,
                    padding: isMissed ? "14px 12px" : "12px",
                    background: isMissed
                      ? "linear-gradient(135deg, #1a0408, #2d0a12)"
                      : `${statusColor}10`,
                    border: `${isMissed ? "2px" : "1px"} solid ${statusColor}${isMissed ? "" : "40"}`,
                    borderRadius: 6, cursor: "pointer", transition: "all 0.2s",
                    boxShadow: isMissed ? `0 0 12px ${statusColor}25` : "none",
                    position: "relative", overflow: "hidden",
                  }}
                  onMouseOver={(e) => { e.currentTarget.style.background = isMissed ? "linear-gradient(135deg,#250812,#3d1020)" : `${statusColor}20`; }}
                  onMouseOut={(e) => { e.currentTarget.style.background = isMissed ? "linear-gradient(135deg,#1a0408,#2d0a12)" : `${statusColor}10`; }}
                >
                  {/* Left accent bar for missed doses */}
                  {isMissed && (
                    <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 4, background: "#ff1744" }} />
                  )}
                  <div style={{ fontSize: 28, flexShrink: 0, marginLeft: isMissed ? 4 : 0 }}>
                    {isMissed ? "🚨" : isTaken ? "✅" : "⏳"}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 14, fontWeight: 700, color: statusColor }}>{med.label.toUpperCase()}</span>
                        {isMissed && (
                          <span style={{ fontSize: 8, fontWeight: 700, letterSpacing: 1, color: "#ff1744", background: "#ff174420", border: "1px solid #ff174460", padding: "1px 6px", borderRadius: 10 }}>MISSED</span>
                        )}
                      </div>
                      <span style={{ fontSize: 12, color: "#c9d1d9", fontWeight: 700 }}>{med.time}</span>
                    </div>
                    <div style={{ fontSize: 10, color: "#8892a4", marginTop: 4, display: "flex", justifyContent: "space-between" }}>
                      <span>
                        {isTaken
                          ? `✅ Taken at ${med.actual}`
                          : isMissed
                          ? `❌ Missed — scheduled for ${med.time}`
                          : `⏳ Upcoming — scheduled for ${med.time}`}
                        {med.meal_relation && med.meal_relation !== "fixed" && (
                          <span style={{
                            marginLeft: 8, fontSize: 9, padding: "1px 6px", borderRadius: 10,
                            background: med.meal_relation === "before" ? "#ff980015" : "#00e67615",
                            border: `1px solid ${med.meal_relation === "before" ? "#ff980040" : "#00e67640"}`,
                            color: med.meal_relation === "before" ? "#ff9800" : "#00e676",
                          }}>
                            {med.meal_relation === "before" ? "Before" : "After"} {med.meal_name || "food"}
                          </span>
                        )}
                      </span>
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
                style={{ width: "100%", background: "#080b12", border: "1px solid #1e2535", color: "#c9d1d9", padding: "10px", borderRadius: 4, marginBottom: 16, outline: "none", fontSize: 12 }} 
              />

              <div style={{ fontSize: 10, color: "#8892a4", marginBottom: 8, letterSpacing: 1 }}>REQUIRED FOR (ILLNESS)</div>
              <input
                type="text"
                value={formIllness}
                onChange={(e) => setFormIllness(e.target.value)}
                placeholder="e.g. Diabetes, Hypertension, Augmentin course"
                style={{ width: "100%", background: "#080b12", border: "1px solid #1e2535", color: "#c9d1d9", padding: "10px", borderRadius: 4, marginBottom: 16, outline: "none", fontSize: 12 }}
              />

              <div style={{ fontSize: 10, color: "#8892a4", marginBottom: 8, letterSpacing: 1 }}>TIME OF DAY</div>
              <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
                {["morning", "afternoon", "night"].map(t => {
                  const selected = formTimes.includes(t);
                  return (
                    <label key={t} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                      <input 
                        type="checkbox" 
                        checked={selected}
                        onChange={() => {
                          setFormTimes(prev => selected ? prev.filter(x => x !== t) : [...prev, t]);
                        }}
                      />
                      <span style={{ fontSize: 12, textTransform: "capitalize" }}>{t}</span>
                    </label>
                  );
                })}
              </div>
              
              {/* ── TAKE WITH FOOD selector ── */}
              <div style={{ fontSize: 10, color: "#8892a4", marginBottom: 8, letterSpacing: 1, marginTop: 8 }}>TAKE WITH FOOD</div>
              <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                {[
                  { val: "fixed",  label: "Fixed Time" },
                  { val: "before", label: "Before Food" },
                  { val: "after",  label: "After Food"  },
                ].map(({ val, label }) => (
                  <button
                    key={val}
                    onClick={() => setFormMealRelation(val)}
                    style={{
                      padding: "6px 14px", borderRadius: 4, fontSize: 10, cursor: "pointer", letterSpacing: 1,
                      background: formMealRelation === val ? (val === "before" ? "#ff980020" : val === "after" ? "#00e67620" : "#4a9eff20") : "transparent",
                      border: `1px solid ${formMealRelation === val ? (val === "before" ? "#ff9800" : val === "after" ? "#00e676" : "#4a9eff") : "#1e2535"}`,
                      color: formMealRelation === val ? (val === "before" ? "#ff9800" : val === "after" ? "#00e676" : "#4a9eff") : "#4a5568",
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {/* ── Meal times (shown when before/after selected) ── */}
              {formMealRelation !== "fixed" && (
                <div style={{ background: "#0a0f1a", border: "1px solid #1e2535", borderRadius: 4, padding: "12px 14px", marginBottom: 16 }}>
                  <div style={{ fontSize: 9, color: "#2d3550", letterSpacing: 2, marginBottom: 10 }}>
                    SET MEAL TIMES — system reminds {formMealRelation === "before" ? "15 min BEFORE" : "RIGHT AFTER"} each selected meal
                  </div>
                  {Object.entries(mealTimes).map(([meal, t]) => (
                    <div key={meal} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                      <span style={{ fontSize: 10, color: "#8892a4", width: 80 }}>{meal}</span>
                      <input
                        type="time"
                        value={t}
                        onChange={(e) => setMealTimes(prev => ({ ...prev, [meal]: e.target.value }))}
                        style={{ background: "#080b12", border: "1px solid #1e2535", color: "#c9d1d9", padding: "4px 8px", borderRadius: 4, fontSize: 11 }}
                      />
                    </div>
                  ))}
                  <div style={{ fontSize: 9, color: "#2d3550", marginTop: 6 }}>Max 3 reminders · every 15 min · stops when pill taken</div>
                </div>
              )}

              <div style={{ display: "flex", gap: 12 }}>
                <button
                  onClick={handleSaveMedicine}
                  style={{ background: "#4a9eff15", border: "1px solid #4a9eff40", color: "#4a9eff", padding: "8px 24px", borderRadius: 4, cursor: "pointer", fontWeight: 700, fontSize: 10, letterSpacing: 1 }}
                >
                  SAVE MEDICINE
                </button>
                <button 
                  onClick={() => { setFormName(""); setFormDose(""); setFormTimes([]); setFormIllness(""); setFormMealRelation("fixed"); }}
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
                // Show — for days with no activity data at all (DB empty / not yet collecting)
                const pillIcon = d.events === 0
                  ? <span style={{ color: "#3a4560", fontSize: 12 }}>—</span>
                  : d.pill ? "✅" : "❌";
                return (
                  <div key={i} style={{ flex: 1, textAlign: "center", background: "#0d1117", padding: "8px 0", borderRadius: 4, border: "1px solid #1e2535" }}>
                    <div style={{ fontSize: 9, color: "#8892a4", marginBottom: 6 }}>{d.day}</div>
                    <div style={{ fontSize: 16 }}>{pillIcon}</div>
                    <div style={{ fontSize: 8, color: d.events === 0 ? "#3a4560" : c, marginTop: 6, fontWeight: 700 }}>{d.events === 0 ? "NO DATA" : `RISK: ${d.risk}`}</div>
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
              <AlertRow key={`alert-${i}`} alert={a} revealed={revealed.includes(i)} />
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
