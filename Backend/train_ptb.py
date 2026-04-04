"""
Train Model 2: PTB binary disease classifier
Classes: 0=Normal Heart, 1=Myocardial Infarction

Run: python train_ptb.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import os

CLASS_NAMES = ["Normal Heart", "Myocardial Infarction"]

# Ensure models folder exists
os.makedirs("models", exist_ok=True)


def load_ptb():
    print("Loading PTB datasets...")

    # ✅ USE BOTH FILES
    ptb_normal   = pd.read_csv("data/ptbdb_normal.csv", header=None)
    ptb_abnormal = pd.read_csv("data/ptbdb_abnormal.csv", header=None)

    # Combine
    combined = pd.concat([ptb_normal, ptb_abnormal], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    X = combined.iloc[:, :-1].values.astype(np.float32)
    y = combined.iloc[:, -1].astype(int).values

    print(f"Original distribution: {Counter(y)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Train: {Counter(y_train)}")
    print(f"Test : {Counter(y_test)}")

    return X_train, y_train, X_test, y_test


def train():
    X_train, y_train, X_test, y_test = load_ptb()

    print("\nTraining Random Forest model...")

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("📊 PTB MODEL PERFORMANCE")
    print("==============================")
    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model (IMPORTANT for FastAPI)
    joblib.dump(clf, "models/model_ptb.pkl")
    print("\n💾 Model saved → models/model_ptb.pkl")


if __name__ == "__main__":
    train()