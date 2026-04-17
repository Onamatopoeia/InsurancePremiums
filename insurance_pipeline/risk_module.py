"""
risk_module.py
--------------
Computes a risk score (0–100) and risk level (Low / Medium / High / Critical)
for a user based on their predicted condition and raw features.
Returns the score, level, and a list of the top contributing risk factors.
"""

import pandas as pd
from typing import Tuple, List

# Base risk points per predicted condition
CONDITION_BASE_RISK = {
    "Healthy":                 5,
    "Asthma":                 20,
    "Depression/Anxiety":     22,
    "Chronic Back Pain":      25,
    "Obesity":                30,
    "Hypertension":           35,
    "Type 2 Diabetes":        45,
    "Cardiovascular Disease": 60,
}

RISK_BANDS = [
    (0,  25,  "Low"),
    (25, 50,  "Medium"),
    (50, 75,  "High"),
    (75, 101, "Critical"),
]


def assess_risk(user: pd.Series) -> Tuple[str, int, List[str]]:
    """
    Assess insurance risk for a single user.

    Parameters
    ----------
    user : pd.Series with all feature columns + 'predicted_condition'

    Returns
    -------
    risk_level   : "Low" | "Medium" | "High" | "Critical"
    risk_score   : int 0–100
    risk_factors : list of human-readable contributing factors
    """
    condition = user.get("predicted_condition", "Healthy")
    score = CONDITION_BASE_RISK.get(condition, 10)
    factors: List[str] = []

    # ── Lifestyle modifiers ──────────────────────────────────────
    if user["smoker"] == 1:
        score += 12
        factors.append("Active smoker (+12 pts)")

    if user["bmi"] >= 35:
        score += 10
        factors.append(f"Severe obesity (BMI {user['bmi']:.1f}, +10 pts)")
    elif user["bmi"] >= 30:
        score += 5
        factors.append(f"Obese (BMI {user['bmi']:.1f}, +5 pts)")

    if user["exercise_days_week"] <= 1:
        score += 7
        factors.append("Sedentary lifestyle (≤1 exercise day/week, +7 pts)")

    if user["alcohol_units_week"] >= 14:
        score += 6
        factors.append(f"High alcohol intake ({user['alcohol_units_week']} units/week, +6 pts)")

    if user["stress_level"] >= 8:
        score += 5
        factors.append(f"High stress (level {user['stress_level']}/10, +5 pts)")

    if user["sleep_hours"] < 6:
        score += 4
        factors.append(f"Sleep deprivation ({user['sleep_hours']:.1f} hrs/night, +4 pts)")

    # ── Clinical markers ─────────────────────────────────────────
    if user["systolic_bp"] >= 160:
        score += 10
        factors.append(f"Severely elevated BP ({user['systolic_bp']}/{user['diastolic_bp']}, +10 pts)")
    elif user["systolic_bp"] >= 140:
        score += 5
        factors.append(f"High BP ({user['systolic_bp']}/{user['diastolic_bp']}, +5 pts)")

    if user["blood_glucose"] >= 200:
        score += 12
        factors.append(f"Critically high blood glucose ({user['blood_glucose']} mg/dL, +12 pts)")
    elif user["blood_glucose"] >= 126:
        score += 6
        factors.append(f"Elevated blood glucose ({user['blood_glucose']} mg/dL, +6 pts)")

    if user["cholesterol"] >= 240:
        score += 6
        factors.append(f"High cholesterol ({user['cholesterol']} mg/dL, +6 pts)")

    # ── History & demographics ────────────────────────────────────
    if user["prior_claims"] >= 5:
        score += 8
        factors.append(f"High prior claims ({user['prior_claims']}, +8 pts)")
    elif user["prior_claims"] >= 2:
        score += 4
        factors.append(f"Prior claims ({user['prior_claims']}, +4 pts)")

    if user["family_history_heart"] == 1:
        score += 5
        factors.append("Family history of heart disease (+5 pts)")

    if user["family_history_diabetes"] == 1:
        score += 4
        factors.append("Family history of diabetes (+4 pts)")

    if user["age"] >= 65:
        score += 8
        factors.append(f"Senior age group ({int(user['age'])} yrs, +8 pts)")
    elif user["age"] >= 50:
        score += 4
        factors.append(f"Age 50+ ({int(user['age'])} yrs, +4 pts)")

    # ── Clamp and classify ────────────────────────────────────────
    score = min(score, 100)

    risk_level = "Low"
    for lo, hi, level in RISK_BANDS:
        if lo <= score < hi:
            risk_level = level
            break

    # Surface top 4 factors only (already ordered by addition sequence ≈ impact)
    top_factors = factors[:4] if factors else ["No significant risk factors identified"]

    return risk_level, score, top_factors
