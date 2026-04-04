"""
FastAPI backend — loads BOTH models at startup, runs both on every /predict call.

Start: uvicorn main:app --reload --port 8000
"""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.preprocessing import preprocess_signal, get_flagged_segment

app = FastAPI(title="CardioAI — Dual Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Class labels ─────────────────────────────────────────────────
RHYTHM_CLASSES  = ["Normal Sinus Rhythm", "Atrial Fibrillation", "Ventricular Fibrillation"]
DISEASE_CLASSES = ["Normal Heart", "Myocardial Infarction"]

# ── Load both models once at startup ────────────────────────────
model_mitbih  = None
model_ptb     = None
USE_CNN       = False

@app.on_event("startup")
def load_models():
    global model_mitbih, model_ptb, USE_CNN

    # Try CNN (.h5) first, fallback to sklearn (.pkl)
    try:
        import tensorflow as tf
        if os.path.exists("models/model_mitbih.h5"):
            model_mitbih = tf.keras.models.load_model("models/model_mitbih.h5")
            USE_CNN = True
            print("✓ Loaded model_mitbih.h5 (CNN)")
        if os.path.exists("models/model_ptb.h5"):
            model_ptb = tf.keras.models.load_model("models/model_ptb.h5")
            print("✓ Loaded model_ptb.h5 (CNN)")
    except Exception:
        pass

    if model_mitbih is None:
        import joblib
        model_mitbih = joblib.load("models/model_mitbih.pkl")
        print("✓ Loaded model_mitbih.pkl (RandomForest)")

    if model_ptb is None:
        import joblib
        model_ptb = joblib.load("models/model_ptb.pkl")
        print("✓ Loaded model_ptb.pkl (RandomForest)")


# ── Request / Response schemas ───────────────────────────────────
class ECGInput(BaseModel):
    signal: list[float]
    fs: int = 360


def run_prediction(model, signal_1d: np.ndarray, class_names: list, use_cnn: bool) -> dict:
    """Run one model and return label + probabilities."""
    if use_cnn:
        x = signal_1d.reshape(1, 187, 1)
        probs = model.predict(x, verbose=0)[0]
    else:
        x = signal_1d.reshape(1, -1)
        probs = model.predict_proba(x)[0]

    pred = int(np.argmax(probs))
    return {
        "prediction":    pred,
        "label":         class_names[pred],
        "confidence":    round(float(probs[pred]) * 100, 1),
        "probabilities": {
            class_names[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probs)
        },
    }


# ── Main predict endpoint ────────────────────────────────────────
@app.post("/predict")
def predict(data: ECGInput):
    if len(data.signal) < 50:
        raise HTTPException(400, "Signal too short — need at least 50 samples")

    # 1. Preprocess (filter, normalize, pad/trim to 187)
    processed = preprocess_signal(data.signal, data.fs)

    # 2. Run BOTH models on the same processed signal
    rhythm_result  = run_prediction(model_mitbih, processed, RHYTHM_CLASSES,  USE_CNN)
    disease_result = run_prediction(model_ptb,    processed, DISEASE_CLASSES, USE_CNN)

    # 3. Find anomalous segment for explainability
    flagged = get_flagged_segment(processed)

    # 4. Compute combined alert level
    alert = _combined_alert(rhythm_result["prediction"], disease_result["prediction"])

    return {
        "rhythm":           rhythm_result,
        "disease":          disease_result,
        "flagged_segment":  flagged,
        "processed_signal": processed.tolist(),
        "alert":            alert,
    }


def _combined_alert(rhythm_pred: int, disease_pred: int) -> dict:
    """Combine both results into one clinical alert level."""
    if rhythm_pred == 2 or disease_pred == 1:
        # VFib OR heart attack → critical
        return {"level": "critical", "color": "#ff4d4d",
                "message": "Critical — immediate intervention required"}
    elif rhythm_pred == 1:
        # AFib → warning
        return {"level": "warning", "color": "#f5c542",
                "message": "Warning — clinical review recommended"}
    else:
        return {"level": "normal", "color": "#00d4aa",
                "message": "Normal — no immediate action required"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "mitbih": model_mitbih is not None,
            "ptb":    model_ptb    is not None,
        }
    }
