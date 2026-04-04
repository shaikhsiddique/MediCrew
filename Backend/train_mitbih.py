"""
Train Model 1: ECG 3-class rhythm classifier

Classes:
0 = Normal Sinus Rhythm
1 = Atrial Fibrillation
2 = Ventricular Fibrillation

Run:
python train_mitbih.py
"""

import numpy as np
import pandas as pd
import joblib
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

print("🚀 Training Random Forest model...")

CLASS_NAMES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Ventricular Fibrillation"
]


# ───────────────────────────────────────────────────────────────
# FIX SIGNAL LENGTH → 187 (VERY IMPORTANT)
# ───────────────────────────────────────────────────────────────
def fix_length(signal, target_len=187):
    if len(signal) > target_len:
        return signal[:target_len]  # trim
    else:
        return np.pad(signal, (0, target_len - len(signal)))  # pad


# ───────────────────────────────────────────────────────────────
# LOAD + PREPROCESS
# ───────────────────────────────────────────────────────────────
def load_data(path: str):
    print("📂 Loading dataset...")

    df = pd.read_csv(path)

    # Convert string → list
    X = df["signal"].apply(ast.literal_eval)
    X = np.array(X.tolist(), dtype=np.float32)

    # ✅ FIX LENGTH HERE
    X = np.array([fix_length(s) for s in X], dtype=np.float32)

    print(f"✅ Signal shape after fix: {X.shape}")

    # Label mapping
    label_map = {
        "Normal": 0,
        "AFib": 1,
        "VFib": 2
    }

    y = df["label"].map(label_map).values

    print(f"📊 Original distribution: {Counter(y)}")

    return X, y


# ───────────────────────────────────────────────────────────────
# TRAIN MODEL
# ───────────────────────────────────────────────────────────────
def train():
    X, y = load_data("data/ecg_dataset_test.csv")

    # ✅ SPLIT FIRST (important)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nBefore SMOTE: {Counter(y_train)}")

    # ✅ SMOTE only on training
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {Counter(y_train)}")
    print(f"Test distribution: {Counter(y_test)}")

    # ── MODEL ──
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    print("\n⚡ Training model...")
    clf.fit(X_train, y_train)

    # ── EVALUATION ──
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("📊 MODEL PERFORMANCE")
    print("==============================")
    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    print("📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── SAVE MODEL ──
    joblib.dump(clf, "models/model_mitbih.pkl")
    print("\n💾 Model saved → models/model_mitbih.pkl")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()