import pandas as pd
import sqlite3
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

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
    (meaning it raised multiple rounds = successful fundraiser)
    Label = 0 if it appears only once
    """
    counts = df["startup_name"].value_counts()
    df["label"] = df["startup_name"].map(lambda x: 1 if counts[x] > 1 else 0)
    print(f"\nLabel distribution:")
    print(f"  Raised multiple rounds (1): {df['label'].sum()}")
    print(f"  Single round only     (0): {(df['label'] == 0).sum()}")
    return df


# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────

def build_features(df):
    """
    Convert categorical columns to numbers using LabelEncoder.
    ML models only understand numbers, not strings.
    """
    df = df.copy()

    # Fill missing values
    df["sector"]          = df["sector"].fillna("Unknown")
    df["city"]            = df["city"].fillna("Unknown")
    df["investment_type"] = df["investment_type"].fillna("Unknown")
    df["amount_usd"]      = df["amount_usd"].fillna(0)

    # Encode categorical columns
    encoders = {}
    for col in ["sector", "city", "investment_type"]:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Final feature set
    features = ["sector_encoded", "city_encoded",
                "investment_type_encoded", "amount_usd"]

    X = df[features]
    y = df["label"]

    print(f"\nFeatures: {features}")
    print(f"Dataset shape: {X.shape}")
    return X, y, encoders


# ── 4. TRAIN MODEL ────────────────────────────────────────────────────────────

def train(X, y):
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining on {len(X_train)} rows, testing on {len(X_test)} rows")

    # Train XGBoost classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model


# ── 5. SAVE MODEL ─────────────────────────────────────────────────────────────

def save_model(model, encoders):
    os.makedirs("Model", exist_ok=True)
    joblib.dump(model,    "Model/model.pkl")
    joblib.dump(encoders, "Model/encoders.pkl")
    print("Saved model to Model/model.pkl")
    print("Saved encoders to Model/encoders.pkl")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df            = load_data()
    df            = create_label(df)
    X, y, encoders = build_features(df)
    model         = train(X, y)
    save_model(model, encoders)