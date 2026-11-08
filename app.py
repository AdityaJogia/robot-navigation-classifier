"""
Streamlit front-end: upload the held-out sensor CSV, pick a trained model,
and see how well it predicts the robot's next move. Models are only loaded
from model/ -- nothing is retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SENSOR_COLS = [f"US{i}" for i in range(1, 25)]
TARGET = "Action"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Robot Wall-Following Classifier", page_icon="🤖", layout="wide")


@st.cache_resource
def load_model(fname):
    return joblib.load(MODEL_DIR / fname)


def score(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    # AUC needs every class present in the uploaded file
    try:
        metrics["AUC"] = roc_auc_score(y, model.predict_proba(X), multi_class="ovr",
                                       average="macro", labels=model.classes_)
    except ValueError:
        metrics["AUC"] = float("nan")
    order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    return {k: metrics[k] for k in order}, y_pred


# ---------------- sidebar ----------------
st.sidebar.title("🤖 Controls")
uploaded = st.sidebar.file_uploader("1. Upload test CSV", type="csv",
                                    help="24 sensor columns US1..US24 plus the 'Action' column")
model_name = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES))

sample = ROOT / "test_data.csv"
if sample.exists():
    st.sidebar.download_button("Download sample test_data.csv", sample.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

# ---------------- header ----------------
st.title("Wall-Following Robot: Navigation Move Classifier")
st.markdown(
    "A SCITOS G5 robot circles a room hugging the wall using **24 ultrasound sensors** "
    "spaced every 15°. Each reading snapshot is labelled with the move it made: "
    "`Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn` or `Slight-Left-Turn`. "
    "Upload the test split to see how each trained classifier performs."
)

if uploaded is None:
    st.info("⬅️ Upload `test_data.csv` from the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in SENSOR_COLS + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {', '.join(missing)}")
    st.stop()

X, y = df[SENSOR_COLS], df[TARGET]

with st.expander(f"Preview uploaded data ({len(df)} rows)"):
    st.dataframe(df.head(20), width="stretch")
    st.bar_chart(y.value_counts())

tab_single, tab_compare = st.tabs([f"📊 {model_name}", "🏁 Compare all models"])

# ---------------- selected model ----------------
with tab_single:
    model = load_model(MODEL_FILES[model_name])
    metrics, y_pred = score(model, X, y)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Confusion matrix")
        labels = list(model.classes_)
        cm = confusion_matrix(y, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels,
                    yticklabels=labels, cbar=False, ax=ax)
        ax.set_xlabel("Predicted move")
        ax.set_ylabel("Actual move")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ---------------- all models side by side ----------------
with tab_compare:
    rows = {name: score(load_model(f), X, y)[0] for name, f in MODEL_FILES.items()}
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d3240"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Best model on this file by MCC: **{best}** ({table.loc[best, 'MCC']:.4f})")
    st.bar_chart(table[["Accuracy", "F1", "MCC"]], stack=False)
