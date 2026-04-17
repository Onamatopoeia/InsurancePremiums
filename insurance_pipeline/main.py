"""
Medical Insurance Risk Assessment Pipeline
==========================================
End-to-end: data → ML → risk scoring → personalized interventions
"""

from data_generator import generate_dataset
from ml_module import train_model, predict_conditions
from risk_module import assess_risk
from intervention_module import generate_interventions
from monitor import simulate_user_activity, retrain_model

import pandas as pd


def run_pipeline(n_train_samples: int = 1000, n_new_users: int = 5):
    print("=" * 60)
    print("  MEDICAL INSURANCE RISK ASSESSMENT PIPELINE")
    print("=" * 60)

    # ── Step 1: Load dynamic dataset ──────────────────────────────
    print("\n[1/5] Generating dynamic dataset...")
    df = generate_dataset(n_samples=n_train_samples)
    print(f"      {len(df)} training records | {df.shape[1]} features")
    print(f"      Columns: {list(df.columns)}")

    # ── Step 2: Train ML model ─────────────────────────────────────
    print("\n[2/5] Training machine learning model...")
    model, feature_cols, label_encoder = train_model(df)
    print(f"      Model: Random Forest  |  Features: {len(feature_cols)}")

    # ── Step 3: Predict conditions for new users ───────────────────
    print("\n[3/5] Predicting medical conditions for new users...")
    new_users = generate_dataset(n_samples=n_new_users, seed=42)
    predictions = predict_conditions(model, new_users[feature_cols], label_encoder)
    new_users["predicted_condition"] = predictions
    print(f"      Processed {n_new_users} new user(s)")

    # ── Step 4: Risk assessment + interventions ────────────────────
    print("\n[4/5] Running risk assessment & generating interventions...")
    print()
    results = []
    for i, (_, user) in enumerate(new_users.iterrows()):
        risk_level, risk_score, risk_factors = assess_risk(user)
        interventions = generate_interventions(risk_level, user, risk_factors)

        result = {
            "user_id": f"USR{i+1:03d}",
            "age": int(user["age"]),
            "predicted_condition": user["predicted_condition"],
            "risk_score": risk_score,
            "risk_level": risk_level,
            "interventions": interventions,
        }
        results.append(result)
        _print_user_report(result)

    # ── Step 5: Monitor activity & retrain ────────────────────────
    print("\n[5/5] Simulating user activity monitoring & model retraining...")
    activity_data = simulate_user_activity(results)
    model = retrain_model(model, df, activity_data, feature_cols)
    print("      Model retrained with activity feedback.\n")

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    return results


def _print_user_report(r: dict):
    bar = "█" * int(r["risk_score"] / 5) + "░" * (20 - int(r["risk_score"] / 5))
    print(f"  ┌─ {r['user_id']}  Age: {r['age']}  Condition: {r['predicted_condition']}")
    print(f"  │  Risk: [{bar}] {r['risk_score']}/100  ({r['risk_level']})")
    print(f"  │  Interventions:")
    for iv in r["interventions"]:
        print(f"  │    • {iv}")
    print(f"  └{'─'*55}")
    print()


if __name__ == "__main__":
    run_pipeline(n_train_samples=2000, n_new_users=5)
