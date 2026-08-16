from pathlib import Path
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
repo = fetch_ucirepo(id=80)
X = repo.data.features.copy()
y = repo.data.targets.copy()
out = X.copy()
out["target"] = y.iloc[:, 0].astype(int).values
out.to_csv(ROOT / "uci_full_dataset.csv", index=False)
print("Saved:", ROOT / "uci_full_dataset.csv", "shape=", out.shape)
