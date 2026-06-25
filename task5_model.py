import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv, os
from datetime import datetime

# ── 0. REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 1. LOAD DATA ─────────────────────────────────────────────────
data = load_iris()
X, y = data.data, data.target

# Train / Validation / Test split (60 / 20 / 20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=SEED)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 2. BASELINE (majority class) ─────────────────────────────────
baseline = DummyClassifier(strategy="most_frequent", random_state=SEED)
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_val, baseline.predict(X_val))
print(f"\nBaseline Accuracy (majority class): {baseline_acc:.4f}")

# ── 3. TRAIN FIRST MODEL ─────────────────────────────────────────
model = DecisionTreeClassifier(max_depth=3, random_state=SEED)
model.fit(X_train, y_train)

# ── 4. EVALUATE ON VALIDATION ────────────────────────────────────
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Model Accuracy on Validation:        {val_acc:.4f}")
print(f"Beats baseline by:                   {val_acc - baseline_acc:+.4f}")

# ── 5. INSPECT WORST ERRORS ──────────────────────────────────────
errors = np.where(val_preds != y_val)[0]
print(f"\nWrong predictions: {len(errors)} / {len(y_val)}")
print("Sample error rows (index | true | predicted):")
for i in errors[:5]:
    print(f"  idx={i}  true={y_val[i]}  pred={val_preds[i]}  features={X_val[i]}")

print("\nClassification Report (Validation):")
print(classification_report(y_val, val_preds, target_names=data.target_names))

# ── 6. EXPERIMENT LOG ────────────────────────────────────────────
log_file = "experiment_log.csv"
write_header = not os.path.exists(log_file)

with open(log_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestamp","model","val_accuracy","baseline_accuracy","delta","notes"])
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "DecisionTree(max_depth=3)",
        round(val_acc, 4),
        round(baseline_acc, 4),
        round(val_acc - baseline_acc, 4),
        "First model run"
    ])

print(f"\nRun logged to {log_file} ✅")