from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE, TEST_SIZE = 73, 0.22

raw = load_digits(as_frame=True).frame
raw.columns = [f"pixel_{i//8}_{i%8}" for i in range(64)] + ["target"]
X, y = raw.drop(columns=["target"]), raw["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": Pipeline([("standardize", StandardScaler()), ("classifier", LogisticRegression(max_iter=3500, C=0.85, solver="lbfgs"))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, criterion="entropy", random_state=RANDOM_STATE),
    "kNN": Pipeline([("standardize", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2))]),
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=420, max_features="sqrt", min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1),
}
def safe_name(n):
    return n.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred, prob = model.predict(X_test), model.predict_proba(X_test)
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }
    joblib.dump(model, MODEL_DIR / f"{safe_name(name)}.joblib")
X_test.assign(target=y_test.values).to_csv(ROOT/"test_data.csv", index=False)
with open(ROOT/"artifacts"/"metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(pd.DataFrame(metrics).T.round(4))
