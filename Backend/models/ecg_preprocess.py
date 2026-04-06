"""
ecg_preprocess.py
Shared preprocessing module -- import in both training AND inference
to guarantee identical pipelines.

Usage in your Flask/FastAPI endpoint:
    from models.ecg_preprocess import predict_ecg
    from tensorflow.keras.models import load_model
    model  = load_model("models/model_mitbih_cnn.keras")
    result = predict_ecg(model, raw_signal_list)
    # result = {"class": "AFib", "confidence": 94.3, ...}
"""
import numpy as np
from scipy.signal import butter, filtfilt

TARGET_LEN  = 187
SAMPLE_RATE = 500
CLASS_NAMES = ["Normal", "AFib", "VFib"]


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=SAMPLE_RATE, order=4):
    nyq  = fs / 2.0
    low  = max(lowcut  / nyq, 1e-4)
    high = min(highcut / nyq, 1 - 1e-4)
    b, a = butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)


def resample_to_target(signal, target=TARGET_LEN):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(signal) == target:
        return signal.astype(np.float32)
    x_new = np.linspace(0, len(signal) - 1, target)
    return np.interp(x_new, np.arange(len(signal)), signal).astype(np.float32)


def normalize(signal):
    signal = np.asarray(signal, dtype=np.float32)
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def prepare(raw_signal, fs=SAMPLE_RATE):
    sig = np.asarray(raw_signal, dtype=np.float64)
    sig = bandpass_filter(sig, fs=fs)
    sig = resample_to_target(sig, TARGET_LEN)
    sig = normalize(sig)
    return sig


def predict_ecg(model, raw_signal, fs=SAMPLE_RATE, threshold=0.70):
    x     = prepare(raw_signal, fs=fs).reshape(1, TARGET_LEN, 1)
    probs = model.predict(x, verbose=0)[0]
    conf  = float(np.max(probs))
    idx   = int(np.argmax(probs))
    return {
        "class":         CLASS_NAMES[idx] if conf >= threshold else "Uncertain",
        "confidence":    round(conf * 100, 2),
        "probabilities": {c: round(float(p) * 100, 2)
                          for c, p in zip(CLASS_NAMES, probs)},
        "uncertain":     conf < threshold,
    }
