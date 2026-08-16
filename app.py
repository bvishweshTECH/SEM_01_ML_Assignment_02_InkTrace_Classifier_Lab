from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

st.set_page_config(page_title="InkTrace Classifier Lab", page_icon="✦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {padding: 1.25rem 1.5rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1.2rem;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><h1>✦ InkTrace Classifier Lab</h1>
<p>Interactive comparison of five classical machine-learning classifiers for handwritten-digit recognition.</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

models = load_models()
with st.sidebar:
    st.header("Experiment controls")
    selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown("**Expected CSV**")
    st.code("64 pixel features + target", language="text")
    st.caption("Upload only the supplied held-out test data.")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload `test_data.csv` to activate the evaluation dashboard.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error("The uploaded CSV must contain a `target` column.")
    st.stop()

feature_columns = [c for c in df.columns if c != TARGET]
if len(feature_columns) != 64:
    st.error(f"Expected 64 pixel features, but found {len(feature_columns)}.")
    st.stop()

model = models[selected_model]
X_input, y_true = df[feature_columns], df[TARGET].astype(int)
pred, proba = model.predict(X_input), model.predict_proba(X_input)

vals = {
    "Accuracy": accuracy_score(y_true, pred),
    "AUC": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
    "Precision": precision_score(y_true, pred, average="weighted", zero_division=0),
    "Recall": recall_score(y_true, pred, average="weighted", zero_division=0),
    "F1": f1_score(y_true, pred, average="weighted", zero_division=0),
    "MCC": matthews_corrcoef(y_true, pred),
}
st.subheader(f"Evaluation — {selected_model}")
cols = st.columns(6)
for col, (label, value) in zip(cols, vals.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(selected_model)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report(y_true, pred, output_dict=True, zero_division=0)).T.round(4), use_container_width=True)

st.subheader("Prediction preview")
preview = X_input.copy()
preview["actual"] = y_true.values
preview["predicted"] = pred
preview["confidence"] = np.max(proba, axis=1)
st.dataframe(preview.head(25).round(4), use_container_width=True)
st.caption("Metrics are recalculated from the uploaded test data.")
