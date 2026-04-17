import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier


def simulate_user_activity(results: List[Dict[str, Any]], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    adherence_by_risk = {
        "Low":      0.85,
        "Medium":   0.65,
        "High":     0.50,
        "Critical": 0.35,
    }

    for r in results:
        base_p = adherence_by_risk.get(r["risk_level"], 0.60)
        n_interventions = len(r["interventions"])
        n_completed = int(rng.binomial(n_interventions, base_p))
        adherence_pct = n_completed / n_interventions if n_interventions else 0

        reduction = adherence_pct * 15 * rng.uniform(0.5, 1.0)
        new_score = max(0, r["risk_score"] - reduction)

        record = {
            "user_id": r["user_id"],
            "initial_risk_level": r["risk_level"],
            "initial_risk_score": r["risk_score"],
            "n_interventions_assigned": n_interventions,
            "n_interventions_completed": n_completed,
            "adherence_pct": round(adherence_pct * 100, 1),
            "follow_up_risk_score": round(new_score, 1),
            "risk_reduced": r["risk_score"] > new_score,
        }
        records.append(record)

        print(
            f"      {r['user_id']}: {n_completed}/{n_interventions} interventions completed "
            f"({adherence_pct:.0%} adherence) → risk score {r['risk_score']} → {new_score:.0f}"
        )

    return pd.DataFrame(records)


def retrain_model(
    model: RandomForestClassifier,
    original_df: pd.DataFrame,
    activity_data: pd.DataFrame,
    feature_cols: List[str],
) -> RandomForestClassifier:
    from ml_module import _encode_features, TARGET_COL
    from sklearn.preprocessing import LabelEncoder

    high_adherence = activity_data[activity_data["adherence_pct"] >= 70]
    n_upsampled = len(high_adherence) * 3

    healthy_records = original_df[original_df[TARGET_COL] == "Healthy"]

    if len(healthy_records) > 0 and n_upsampled > 0:
        extra = healthy_records.sample(
            n=min(n_upsampled, len(healthy_records)),
            replace=True,
            random_state=1,
        )
        augmented_df = pd.concat([original_df, extra], ignore_index=True)
    else:
        augmented_df = original_df.copy()

    df_enc, _ = _encode_features(augmented_df)
    X = df_enc[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df_enc[TARGET_COL].values)

    model.set_params(warm_start=False)
    model.fit(X, y)

    print(
        f"      Retrained on {len(augmented_df)} records "
        f"({len(augmented_df) - len(original_df)} activity-derived samples added)"
    )
    return model
