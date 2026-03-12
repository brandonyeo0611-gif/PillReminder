"""
llm_explainer.py
=================
Calls Groq (free tier) to produce a plain-English explanation of a risk result.
Falls back to a structured default if Groq is unavailable or key is invalid.

Public interface:
    explain_risk(person_id, risk_score, risk_level, anomalies, rag_context) -> dict

Return shape (always present, even on fallback):
    {
        "summary":       str,   # 2 sentences for family, no markdown
        "concern_level": str,   # "normal" | "watch" | "urgent"
        "action":        str,   # one specific thing family should do now
        "positive":      str,   # one positive observation about today
    }

Requires env var: GROQ_API_KEY
Set in .env file or: export GROQ_API_KEY="gsk_..."
"""

import os
import json
from pathlib import Path

# Load .env from repo root if present
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass  # dotenv not installed; key must be set via export

try:
    from groq import Groq
    _groq_available = True
except ImportError:
    print("groq not installed. Run: pip install groq")
    _groq_available = False

# concern_level map used by _fallback
_LEVEL_TO_CONCERN = {
    "GREEN":   "normal",
    "YELLOW":  "watch",
    "RED":     "urgent",
    "UNKNOWN": "watch",
}


def explain_risk(
    person_id: str,
    risk_score: int,
    risk_level: str,
    anomalies: list,
    rag_context: str = "",
) -> dict:
    """
    Call Groq to explain a risk result in plain English for a family member.
    Always returns a dict with keys: summary, concern_level, action, positive.
    Never raises — returns _fallback() on any failure.

    anomalies: list of dicts from deviation_detector.check() — strings are tolerated
    rag_context: plain string from RAGRetriever.get_context() — empty string is fine
    """
    # Check key at call time (not import time) so missing key → fallback, not crash
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not _groq_available or not api_key:
        return _fallback(risk_score, risk_level, anomalies)

    # Filter anomalies to dicts only for clean JSON serialisation
    clean_anomalies = [a for a in anomalies if isinstance(a, dict)]

    context_block = (
        f"\nMedical context from knowledge base:\n{rag_context}"
        if rag_context else ""
    )

    prompt = f"""You are CareWatch, a caring elderly monitoring assistant.
A family member is checking on their loved one. Be warm, clear, and concise.
Do not use markdown, asterisks, or underscores anywhere in your response.

Return ONLY valid JSON with exactly these four keys:
{{
  "summary": "2 sentences explaining what happened today in plain English",
  "concern_level": "normal or watch or urgent",
  "action": "one specific thing the family should do right now",
  "positive": "one positive observation about today"
}}

Data:
- Person: {person_id}
- Risk Score: {risk_score}/100
- Risk Level: {risk_level}
- Issues detected: {json.dumps(clean_anomalies)}{context_block}

JSON only. No markdown. No extra text. No explanation outside the JSON."""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # llama3-8b-8192 decommissioned; this is the replacement
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps response anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        # Confirm all 4 keys present — if any missing, use fallback
        required = {"summary", "concern_level", "action", "positive"}
        if not required.issubset(parsed.keys()):
            print(f"LLM response missing keys: {required - parsed.keys()}. Using fallback.")
            return _fallback(risk_score, risk_level, anomalies)

        # Normalise concern_level to known values
        parsed["concern_level"] = parsed["concern_level"].lower().strip()
        if parsed["concern_level"] not in ("normal", "watch", "urgent"):
            parsed["concern_level"] = _LEVEL_TO_CONCERN.get(risk_level, "watch")

        return parsed

    except json.JSONDecodeError as e:
        print(f"LLM returned non-JSON: {e}. Using fallback.")
        return _fallback(risk_score, risk_level, anomalies)
    except Exception as e:
        print(f"LLM call failed: {e}. Using fallback.")
        return _fallback(risk_score, risk_level, anomalies)


def _fallback(risk_score: int, risk_level: str, anomalies: list) -> dict:
    """
    Returns a valid explanation dict when Groq is unavailable.
    Always has all 4 keys. Never raises.
    """
    n = len([a for a in anomalies if isinstance(a, dict)])
    return {
        "summary": (
            f"Risk score is {risk_score}/100 ({risk_level}). "
            f"{n} issue(s) detected today."
        ),
        "concern_level": _LEVEL_TO_CONCERN.get(risk_level, "watch"),
        "action": (
            "Call or visit immediately."
            if risk_level == "RED"
            else "Check the CareWatch dashboard for details."
        ),
        "positive": "Monitoring is active and working normally.",
    }
