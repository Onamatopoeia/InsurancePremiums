"""
data_generator.py
-----------------
Generates a realistic synthetic dataset of insurance users.
Features: demographics, lifestyle, medical history, insurance history.
"""

import numpy as np
import pandas as pd
from typing import Optional


CONDITIONS = [
    "Healthy",
    "Type 2 Diabetes",
    "Hypertension",
    "Cardiovascular Disease",
    "Obesity",
    "Asthma",
    "Depression/Anxiety",
    "Chronic Back Pain",
]

CONDITION_WEIGHTS = [0.35, 0.15, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03]


def generate_dataset(n_samples: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate a synthetic user dataset for training or inference."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, size=n_samples)
    bmi = rng.normal(loc=27, scale=5, size=n_samples).clip(15, 50)
    systolic_bp = rng.normal(loc=120, scale=15, size=n_samples).clip(80, 200)
    diastolic_bp = rng.normal(loc=80, scale=10, size=n_samples).clip(50, 130)
    cholesterol = rng.normal(loc=200, scale=35, size=n_samples).clip(120, 320)
    blood_glucose = rng.normal(loc=95, scale=20, size=n_samples).clip(60, 300)

    smoker = rng.binomial(1, 0.18, size=n_samples)
    alcohol_units_week = rng.integers(0, 30, size=n_samples)
    exercise_days_week = rng.integers(0, 7, size=n_samples)
    sleep_hours = rng.normal(loc=7, scale=1.2, size=n_samples).clip(3, 12)
    stress_level = rng.integers(1, 11, size=n_samples)  # 1–10

    prior_claims = rng.integers(0, 10, size=n_samples)
    family_history_heart = rng.binomial(1, 0.25, size=n_samples)
    family_history_diabetes = rng.binomial(1, 0.20, size=n_samples)
    employment_status = rng.choice(
        ["employed", "self_employed", "unemployed", "retired"], size=n_samples,
        p=[0.55, 0.15, 0.10, 0.20]
    )
    income_bracket = rng.choice(
        ["low", "medium", "high"], size=n_samples, p=[0.30, 0.45, 0.25]
    )

    # Derive condition with some feature correlation
    condition_probs = _compute_condition_probs(
        rng, n_samples, age, bmi, systolic_bp, blood_glucose,
        smoker, family_history_heart, family_history_diabetes
    )
    condition = [
        rng.choice(CONDITIONS, p=p) for p in condition_probs
    ]

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi.round(1),
        "systolic_bp": systolic_bp.round(0).astype(int),
        "diastolic_bp": diastolic_bp.round(0).astype(int),
        "cholesterol": cholesterol.round(0).astype(int),
        "blood_glucose": blood_glucose.round(0).astype(int),
        "smoker": smoker,
        "alcohol_units_week": alcohol_units_week,
        "exercise_days_week": exercise_days_week,
        "sleep_hours": sleep_hours.round(1),
        "stress_level": stress_level,
        "prior_claims": prior_claims,
        "family_history_heart": family_history_heart,
        "family_history_diabetes": family_history_diabetes,
        "employment_status": employment_status,
        "income_bracket": income_bracket,
        "condition": condition,
    })

    return df


def _compute_condition_probs(
    rng, n, age, bmi, sbp, glucose, smoker, fh_heart, fh_diabetes
) -> list:
    """Adjust condition probabilities based on risk factors."""
    probs = []
    base = np.array(CONDITION_WEIGHTS, dtype=float)

    for i in range(n):
        w = base.copy()

        # Diabetes risk
        if glucose[i] > 126 or bmi[i] > 30 or fh_diabetes[i]:
            w[1] *= 3.0
        # Hypertension
        if sbp[i] > 140 or age[i] > 55:
            w[2] *= 2.5
        # Cardiovascular
        if smoker[i] or fh_heart[i] or age[i] > 60:
            w[3] *= 2.5
        # Obesity
        if bmi[i] > 30:
            w[4] *= 3.0
        # Healthy less likely if multiple risks
        risk_count = sum([
            glucose[i] > 126, bmi[i] > 30, sbp[i] > 140,
            smoker[i], fh_heart[i], age[i] > 60
        ])
        w[0] *= max(0.1, 1.0 - risk_count * 0.2)

        w /= w.sum()
        probs.append(w)

    return probs
