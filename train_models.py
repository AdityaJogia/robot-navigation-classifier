"""
Train five classifiers on the UCI Wall-Following Robot Navigation dataset
(24 ultrasound sensors -> 4 movement commands) and save them to model/.

Run:  python train_models.py
Outputs:
    model/<name>.joblib      one fitted pipeline (scaler + classifier) per model
    model/metrics.json       test-set metrics for every model
    test_data.csv            held-out 20% split (features + target) for the app
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "sensor_readings_24.data"
MODEL_DIR = ROOT / "model"
SEED = 26

# US1 points straight ahead (180 deg); the rest step round the robot in 15 deg increments
SENSOR_COLS = [f"US{i}" for i in range(1, 25)]
TARGET = "Action"


def load_data():
    df = pd.read_csv(RAW_FILE, header=None, names=SENSOR_COLS + [TARGET])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_models():
    # Every model is wrapped with a scaler so the app can feed raw sensor values straight in.
    # Scaling barely matters for the tree / NB but keeps the saved artefacts uniform.
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2,
                                                random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=10, gamma="scale",
                                      probability=True, random_state=SEED),
    }


def evaluate(pipe, X, y):
    """Six required metrics. Multi-class -> macro averages and one-vs-rest AUC."""
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="macro",
                             labels=pipe.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def slug(name):
    return name.lower().replace(" ", "_")


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {len(SENSOR_COLS)} features")
    print(df[TARGET].value_counts().to_string(), "\n")

    X, y = df[SENSOR_COLS], df[TARGET]
    # stratify keeps the rare Slight-Left-Turn class in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    all_metrics = {}
    for name, clf in build_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / f"{slug(name)}.joblib")
        all_metrics[name] = {k: round(float(v), 4) for k, v in evaluate(pipe, X_test, y_test).items()}
        print(f"{name:<24}", "  ".join(f"{k}={v:.4f}" for k, v in all_metrics[name].items()))

    (MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    table = pd.DataFrame(all_metrics).T
    print("\nMarkdown table for README:\n")
    print("| ML Model Name | " + " | ".join(table.columns) + " |")
    print("|---" * (len(table.columns) + 1) + "|")
    for name, row in table.iterrows():
        print(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")


if __name__ == "__main__":
    main()
