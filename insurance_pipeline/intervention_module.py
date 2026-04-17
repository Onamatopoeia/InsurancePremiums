
import pandas as pd
from typing import List


def generate_interventions(
    risk_level: str,
    user: pd.Series,
    risk_factors: List[str],
) -> List[str]:
    interventions: List[str] = []
    condition = user.get("predicted_condition", "Healthy")

    condition_programs = {
        "Type 2 Diabetes": (
            "Enroll in Diabetes Prevention Program (DPP) — "
            "structured diet, medication adherence & glucose monitoring"
        ),
        "Hypertension": (
            "Join Hypertension Management Program — "
            "DASH diet coaching, BP self-monitoring kit, and medication review"
        ),
        "Cardiovascular Disease": (
            "Refer to Cardiac Rehabilitation Program — "
            "supervised exercise, nutritional counselling, and cardiology follow-ups"
        ),
        "Obesity": (
            "Enrol in Medical Weight Management Program — "
            "dietitian consultations, behavioral therapy & 12-week exercise plan"
        ),
        "Asthma": (
            "Asthma Action Plan — personalised trigger avoidance guide, "
            "inhaler technique check, and pulmonologist telehealth session"
        ),
        "Depression/Anxiety": (
            "Mental Wellness Program — access to licensed therapist (8 sessions), "
            "mindfulness app subscription, and peer support group"
        ),
        "Chronic Back Pain": (
            "Back Health Program — physiotherapy (6 sessions), "
            "ergonomics assessment, and low-impact exercise plan"
        ),
    }

    if condition in condition_programs:
        interventions.append(condition_programs[condition])

    if user["smoker"] == 1:
        interventions.append(
            "Smoking Cessation Program — nicotine replacement therapy, "
            "counselling hotline access & premium reduction upon 12-month abstinence"
        )

    if user["exercise_days_week"] <= 1:
        interventions.append(
            "Physical Activity Initiative — personalised 8-week starter plan "
            "with weekly check-ins; gym membership subsidy available"
        )
    elif user["exercise_days_week"] <= 3:
        interventions.append(
            "Exercise Enhancement — increase to ≥150 min moderate activity/week; "
            "wearable device incentive programme available"
        )

    if user["bmi"] >= 30:
        interventions.append(
            "Nutrition Coaching — 1:1 sessions with registered dietitian; "
            "meal-planning app (3 months free)"
        )

    if user["alcohol_units_week"] >= 14:
        interventions.append(
            "Alcohol Reduction Support — brief intervention counselling "
            "and confidential helpline (free, 24/7)"
        )

    if user["stress_level"] >= 7:
        interventions.append(
            "Stress Management Program — 6-week mindfulness-based stress reduction "
            "(MBSR) course, available online"
        )

    if user["sleep_hours"] < 6:
        interventions.append(
            "Sleep Health Program — sleep hygiene coaching and screening "
            "for sleep disorders (covered under wellness benefit)"
        )

    if user["blood_glucose"] >= 100:
        interventions.append(
            "Annual HbA1c & fasting glucose screening — "
            "early detection reduces diabetes progression risk by up to 58%"
        )

    if user["systolic_bp"] >= 130:
        interventions.append(
            "Home BP monitoring kit — provided at no cost; "
            "monthly readings reported to care coordinator"
        )

    if user["cholesterol"] >= 200:
        interventions.append(
            "Lipid Management — annual lipid panel + dietary counselling; "
            "statin therapy referral if clinically indicated"
        )

    if user["age"] >= 50:
        interventions.append(
            "Preventive Screenings Package — age-appropriate cancer screenings, "
            "bone density scan, and annual cardiovascular risk assessment"
        )

    if user["family_history_heart"] == 1 or user["family_history_diabetes"] == 1:
        interventions.append(
            "Genetic & Family Risk Counselling — understand inherited risks "
            "and get a proactive surveillance plan"
        )

    if risk_level == "Critical":
        interventions.insert(
            0,
            "⚠️  URGENT: Assigned to a dedicated Care Navigator for "
            "immediate review and care coordination within 48 hours"
        )
    elif risk_level == "High":
        interventions.insert(
            0,
            "Priority Health Review — scheduled with in-network physician "
            "within 2 weeks (co-pay waived)"
        )


    if risk_level in ("Medium", "High", "Critical"):
        interventions.append(
            "Premium Reduction Pathway — complete 3+ interventions within "
            "6 months to qualify for up to 15% annual premium discount"
        )

    return interventions[:6]
