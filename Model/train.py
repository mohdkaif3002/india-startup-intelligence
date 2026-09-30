import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_data():
    conn = sqlite3.connect("Data/startups.db")
    df = pd.read_sql("SELECT * FROM startups", conn)
    conn.close()
    print(f"Loaded {len(df)} rows from database")
    return df


# ── 2. CREATE LABEL ───────────────────────────────────────────────────────────

def create_label(df):
    """
    Label = 1 if this startup appears more than once in the dataset
    (proxy for "raised multiple rounds"), else 0.
    NOTE: this measures repeat appearance in the data, not fundraising success.
    """
    counts = df["startup_name"].value_counts()
    df["label"] = df["startup_name"].map(lambda x: 1 if counts[x] > 1 else 0)
    print("\nLabel distribution:")
    print(f"  Multiple appearances (1): {df['label'].sum()}")
    print(f"  Single appearance    (0): {(df['label'] == 0).sum()}")
    return df


# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────

def build_features(df):
    """Same features and encoders as before, so Backend/main.py keeps working."""
    df = df.copy()

    df["sector"] = df["sector"].fillna("Unknown")
    df["city"] = df["city"].fillna("Unknown")
    df["investment_type"] = df["investment_type"].fillna("Unknown")
    df["amount_usd"] = df["amount_usd"].fillna(0)

    encoders = {}
    for col in ["sector", "city", "investment_type"]:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    features = ["sector_encoded", "city_encoded",
                "investment_type_encoded", "amount_usd"]

    X = df[features]
    y = df["label"]
    groups = df["startup_name"]  # used to keep each startup on one side of the split

    print(f"\nFeatures: {features}")
    print(f"Dataset shape: {X.shape}")
    return X, y, encoders, groups


# ── 4. EVALUATION HELPERS ─────────────────────────────────────────────────────

def make_model(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    both_classes = y_test.nunique() == 2
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4) if both_classes else None,
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4) if both_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def print_metrics(title, m):
    print(f"\n{title}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"  {k:<10}: {m[k]}")
    print(f"  confusion : {m['confusion_matrix']}")


# ── 5. TRAIN + COMPARE ────────────────────────────────────────────────────────

def train(X, y, groups):
    results = {}

    # A) Row-level random split (what the original script did).
    #    Rows from the SAME startup can land in both train and test.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m_random = make_model(spw).fit(Xtr, ytr)
    results["random_split"] = evaluate(m_random, Xte, yte)

    # B) Group-aware split: every startup is entirely in train OR test.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    Xtr_g, Xte_g = X.iloc[tr_idx], X.iloc[te_idx]
    ytr_g, yte_g = y.iloc[tr_idx], y.iloc[te_idx]
    assert set(groups.iloc[tr_idx]).isdisjoint(set(groups.iloc[te_idx]))

    print(f"\nGroup split -> train {len(Xtr_g)} rows, test {len(Xte_g)} rows")
    print(f"Positive rate in test set: {yte_g.mean():.2%}")

    baseline = DummyClassifier(strategy="most_frequent").fit(Xtr_g, ytr_g)
    results["baseline_majority_class"] = evaluate(baseline, Xte_g, yte_g)

    spw_g = (ytr_g == 0).sum() / max((ytr_g == 1).sum(), 1)
    model = make_model(spw_g).fit(Xtr_g, ytr_g)
    results["group_split"] = evaluate(model, Xte_g, yte_g)

    print_metrics("Baseline (always predict majority class), group split", results["baseline_majority_class"])
    print_metrics("XGBoost, random row split (may leak)", results["random_split"])
    print_metrics("XGBoost, group split (honest estimate)", results["group_split"])

    importances = dict(zip(X.columns, [round(float(v), 4) for v in model.feature_importances_]))
    results["feature_importance"] = importances
    print(f"\nFeature importance: {importances}")

    return model, results


# ── 6. SAVE ───────────────────────────────────────────────────────────────────

def save_model(model, encoders, results):
    os.makedirs("Model", exist_ok=True)
    joblib.dump(model, "Model/model.pkl")
    joblib.dump(encoders, "Model/encoders.pkl")
    with open("Model/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved Model/model.pkl, Model/encoders.pkl, Model/metrics.json")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    df = create_label(df)
    X, y, encoders, groups = build_features(df)
    model, results = train(X, y, groups)
    save_model(model, encoders, results)
