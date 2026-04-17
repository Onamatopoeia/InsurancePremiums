"""
ml_module.py
------------
Trains a Random Forest classifier on the dynamic dataset to predict
medical conditions. Returns the model, feature list, and label encoder.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Tuple, List


CATEGORICAL_COLS = ["employment_status", "income_bracket"]
TARGET_COL = "condition"


def _encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode categorical columns."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df = df.copy()
    df[CATEGORICAL_COLS] = enc.fit_transform(df[CATEGORICAL_COLS])
    return df, enc


def train_model(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[RandomForestClassifier, List[str], LabelEncoder]:
    """
    Train a RandomForest on the dataset.

    Returns
    -------
    model         : trained classifier
    feature_cols  : ordered list of feature column names
    label_encoder : fitted LabelEncoder for the target
    """
    df_enc, _ = _encode_features(df)

    feature_cols = [c for c in df_enc.columns if c != TARGET_COL]
    X = df_enc[feature_cols].values
    y_raw = df_enc[TARGET_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    if verbose:
        acc = model.score(X_test, y_test)
        print(f"      Validation accuracy: {acc:.2%}")
        report = classification_report(
            y_test, model.predict(X_test),
            target_names=le.classes_, zero_division=0
        )
        # Print a condensed version
        lines = [l for l in report.split("\n") if l.strip()]
        print(f"      {'Class':<30} {'Precision':>9} {'Recall':>7} {'F1':>6}")
        print(f"      {'─'*55}")
        for line in lines[1:-3]:
            parts = line.split()
            if len(parts) >= 5:
                label = " ".join(parts[:-4])
                p, r, f = parts[-4], parts[-3], parts[-2]
                print(f"      {label:<30} {p:>9} {r:>7} {f:>6}")

    return model, feature_cols, le


def predict_conditions(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> List[str]:
    """
    Predict medical conditions for new user records.
    X must contain only the feature columns (no target).
    """
    df_enc, _ = _encode_features(X.assign(condition="Healthy"))
    feature_cols = [c for c in df_enc.columns if c != TARGET_COL]
    X_vals = df_enc[feature_cols].values
    preds = model.predict(X_vals)
    return label_encoder.inverse_transform(preds).tolist()
