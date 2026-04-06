"""
train_from_csv.py -- Production-ready 3-class ECG ResNet
========================================================
Classes : Normal (0), AFib (1), VFib (2)
Input   : ecg_dataset.csv  (columns: label, signal)
Output  : models/model_mitbih_cnn.keras
          models/ecg_preprocess.py   (shared preprocessing module)

Improvements over the basic CNN
---------------------------------
1. ResNet architecture  -- residual blocks, GlobalAveragePooling
2. Bandpass filter      -- removes baseline wander & HF noise (scipy)
3. Realistic augmentation -- baseline wander, powerline noise, spikes,
                             DC offset, amplitude scaling, time shift
4. Harder dropout / L2 regularization to reduce overfitting
5. Label smoothing loss -- prevents overconfident predictions
6. Confidence thresholding at inference
7. SyntaxWarning fix    -- forward-slash paths
8. Unicode fix          -- encoding="utf-8" when saving preprocess module

Run:
    pip install scipy
    python train_from_csv.py
"""

import os
import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    Dense, Add, Input, GlobalAveragePooling1D, Activation
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from scipy.signal import butter, filtfilt

# -- Config -------------------------------------------------------------------
CLASS_NAMES = ["Normal", "AFib", "VFib"]
LABEL_MAP   = {"Normal": 0, "AFib": 1, "VFib": 2}
TARGET_LEN  = 187
SAMPLE_RATE = 500          # Hz -- assumed for your dataset
CSV_PATH    = "data/ecg_dataset_test.csv"   # change path if needed
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "model_mitbih_cnn.keras")
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# 1. PREPROCESSING  (identical pipeline used at training AND inference)
# =============================================================================

def bandpass_filter(signal: np.ndarray,
                    lowcut: float = 0.5,
                    highcut: float = 40.0,
                    fs: int = SAMPLE_RATE,
                    order: int = 4) -> np.ndarray:
    """Remove baseline wander (< 0.5 Hz) and high-freq noise (> 40 Hz)."""
    nyq  = fs / 2.0
    low  = max(lowcut  / nyq, 1e-4)
    high = min(highcut / nyq, 1 - 1e-4)
    b, a = butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)


def resample_to_target(signal: np.ndarray, target: int = TARGET_LEN) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(signal) == target:
        return signal.astype(np.float32)
    x_new = np.linspace(0, len(signal) - 1, target)
    return np.interp(x_new, np.arange(len(signal)), signal).astype(np.float32)


def normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def prepare(raw_signal, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Full preprocessing pipeline: filter -> resample -> normalize."""
    sig = np.asarray(raw_signal, dtype=np.float64)
    sig = bandpass_filter(sig, fs=fs)
    sig = resample_to_target(sig, TARGET_LEN)
    sig = normalize(sig)
    return sig


# =============================================================================
# 2. REALISTIC AUGMENTATION
# =============================================================================

def augment_realistic(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Physics-based ECG augmentation simulating real-world signal corruption.
    Each transform applied probabilistically for diverse training samples.
    """
    x = x.copy().astype(np.float64)
    t = np.arange(len(x))
    t_norm = t / len(x)

    # Amplitude scaling
    x *= rng.uniform(0.7, 1.3)

    # Baseline wander (mimics respiration ~0.05-0.4 Hz)
    if rng.random() < 0.8:
        bw_freq  = rng.uniform(0.05, 0.4)
        bw_amp   = rng.uniform(0.05, 0.35)
        bw_phase = rng.uniform(0, 2 * np.pi)
        x += bw_amp * np.sin(2 * np.pi * bw_freq * t_norm + bw_phase)

    # Power line interference (50 Hz or 60 Hz)
    if rng.random() < 0.5:
        pl_freq_hz = float(rng.choice([50, 60]))
        pl_freq    = pl_freq_hz / SAMPLE_RATE
        pl_amp     = rng.uniform(0.01, 0.08)
        x += pl_amp * np.sin(2 * np.pi * pl_freq * t)

    # Gaussian muscle / EMG noise
    if rng.random() < 0.9:
        noise_std = rng.uniform(0.02, 0.15)
        x += rng.normal(0, noise_std, size=x.shape)

    # DC offset
    if rng.random() < 0.4:
        x += rng.uniform(-0.3, 0.3)

    # Random time shift
    if rng.random() < 0.7:
        x = np.roll(x, rng.integers(-25, 26))

    # Electrode spike artifact
    if rng.random() < 0.15:
        n_spikes = rng.integers(1, 4)
        for _ in range(n_spikes):
            pos = rng.integers(0, len(x))
            x[pos] += rng.uniform(1.5, 4.0) * float(rng.choice([-1, 1]))

    # Segment dropout (simulate lead-off)
    if rng.random() < 0.1:
        start  = rng.integers(0, len(x) - 10)
        length = rng.integers(5, 20)
        x[start:start + length] = 0.0

    return normalize(x.astype(np.float32))


def oversample_with_augmentation(X: np.ndarray,
                                  y: np.ndarray,
                                  seed: int = 42) -> tuple:
    """Augment minority classes up to majority count using realistic noise."""
    rng      = np.random.default_rng(seed)
    counts   = Counter(y.tolist())
    majority = counts.most_common(1)[0][1]
    X_out, y_out = [X.copy()], [y.copy()]
    for cls, n_have in counts.items():
        n_need = majority - n_have
        if n_need <= 0:
            continue
        print(f"  class {cls} ({CLASS_NAMES[cls]}): {n_have} real -> +{n_need} augmented")
        idx    = np.where(y == cls)[0]
        chosen = idx[rng.integers(0, n_have, size=n_need)]
        aug    = np.stack([augment_realistic(X[i], rng) for i in chosen])
        X_out.append(aug)
        y_out.append(np.full(n_need, cls, dtype=y.dtype))
    X_all = np.concatenate(X_out)
    y_all = np.concatenate(y_out)
    perm  = rng.permutation(len(X_all))
    return X_all[perm], y_all[perm]


# =============================================================================
# 3. RESNET ARCHITECTURE
# =============================================================================

def residual_block(x, filters: int, kernel_size: int = 3,
                   downsample: bool = False, reg: float = 1e-4):
    """
    Pre-activation residual block (He et al.).
    downsample=True halves the temporal dimension via MaxPooling.
    """
    shortcut = x

    # Main path
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, kernel_size, padding="same",
               kernel_regularizer=l2(reg))(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, kernel_size, padding="same",
               kernel_regularizer=l2(reg))(x)

    if downsample:
        x = MaxPooling1D(2, padding="same")(x)

    # Shortcut projection
    if shortcut.shape[-1] != filters or downsample:
        shortcut = Conv1D(filters, 1, padding="same",
                          kernel_regularizer=l2(reg))(shortcut)
        if downsample:
            shortcut = MaxPooling1D(2, padding="same")(shortcut)

    return Add()([x, shortcut])


def build_resnet(target_len: int = TARGET_LEN,
                 n_classes: int = 3,
                 reg: float = 1e-4) -> Model:
    """
    1-D ResNet for ECG classification.
    GlobalAveragePooling replaces the large Flatten->Dense bottleneck.
    Label smoothing in the loss combats overconfidence on clean data.
    """
    inputs = Input(shape=(target_len, 1), name="ecg_input")

    # Stem
    x = Conv1D(64, 7, padding="same", kernel_regularizer=l2(reg),
               name="stem_conv")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(2, padding="same")(x)        # 187 -> ~94

    # Stage 1: 64 filters, no downsample
    x = residual_block(x, 64,  kernel_size=5, downsample=False, reg=reg)
    x = Dropout(0.2)(x)

    # Stage 2: 128 filters, downsample
    x = residual_block(x, 128, kernel_size=5, downsample=True,  reg=reg)
    x = Dropout(0.2)(x)

    # Stage 3: 256 filters, downsample
    x = residual_block(x, 256, kernel_size=3, downsample=True,  reg=reg)
    x = Dropout(0.3)(x)

    # Stage 4: 256 filters, no downsample
    x = residual_block(x, 256, kernel_size=3, downsample=False, reg=reg)
    x = Dropout(0.3)(x)

    # Head
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(reg))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="ECG_ResNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# 4. INFERENCE HELPER
# =============================================================================

def predict_ecg(model, raw_signal, fs: int = SAMPLE_RATE,
                threshold: float = 0.70) -> dict:
    """
    Predict ECG class with confidence gating.

    Parameters
    ----------
    model      : loaded Keras model
    raw_signal : list or np.ndarray of raw ECG samples
    fs         : sampling rate of raw_signal (default 500 Hz)
    threshold  : predictions below this confidence return 'Uncertain'

    Returns
    -------
    dict: class, confidence (%), per-class probabilities (%), uncertain flag
    """
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


# =============================================================================
# 5. TRAINING PIPELINE
# =============================================================================

def train():
    # Load
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(f"  Class distribution: {dict(Counter(df['label'].tolist()))}")

    # Parse & preprocess
    print("\nParsing & preprocessing (bandpass -> resample -> normalize)...")
    X_list, y_list = [], []
    for i, row in df.iterrows():
        raw = (ast.literal_eval(row["signal"])
               if isinstance(row["signal"], str)
               else list(row["signal"]))
        X_list.append(prepare(raw))
        y_list.append(LABEL_MAP[row["label"]])
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)} processed ...")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int32)
    print(f"  X shape: {X.shape}  |  y: {dict(Counter(y.tolist()))}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)}  |  Test: {len(y_test)}")

    # Augment
    print("\nAugmenting with realistic noise...")
    X_train, y_train = oversample_with_augmentation(X_train, y_train)
    print(f"  After augmentation: {dict(Counter(y_train.tolist()))}")

    # Reshape + one-hot
    X_train_r = X_train.reshape(-1, TARGET_LEN, 1)
    X_test_r  = X_test.reshape(-1, TARGET_LEN, 1)
    y_train_c = to_categorical(y_train, 3)
    y_test_c  = to_categorical(y_test,  3)

    # Build
    model = build_resnet()
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print("\nTraining ECG ResNet...")
    model.fit(
        X_train_r, y_train_c,
        validation_split=0.15,
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_proba = model.predict(X_test_r, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    acc          = accuracy_score(y_test, y_pred)
    confidences  = np.max(y_pred_proba, axis=1)
    uncertain    = int((confidences < 0.70).sum())

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE  (test set)")
    print("=" * 60)
    print(f"Overall accuracy      : {acc * 100:.2f}%")
    print(f"Mean confidence       : {confidences.mean() * 100:.1f}%")
    print(f"Uncertain predictions : {uncertain}/{len(y_test)} "
          f"({uncertain / len(y_test) * 100:.1f}% below 70% threshold)\n")

    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("Confusion matrix (rows=true, cols=pred):")
    print(f"{'':12s}  {'Normal':>8}  {'AFib':>8}  {'VFib':>8}")
    cm = confusion_matrix(y_test, y_pred)
    for i, row_vals in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:10s}  "
              f"{row_vals[0]:>8}  {row_vals[1]:>8}  {row_vals[2]:>8}")

    # Save model
    model.save(MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    # Save shared preprocessing module
    # FIX: encoding="utf-8" prevents UnicodeEncodeError on Windows (cp1252)
    preprocess_code = """\
\"\"\"
ecg_preprocess.py
Shared preprocessing module -- import in both training AND inference
to guarantee identical pipelines.

Usage in your Flask/FastAPI endpoint:
    from models.ecg_preprocess import predict_ecg
    from tensorflow.keras.models import load_model
    model  = load_model("models/model_mitbih_cnn.keras")
    result = predict_ecg(model, raw_signal_list)
    # result = {"class": "AFib", "confidence": 94.3, ...}
\"\"\"
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
"""
    preprocess_path = os.path.join(MODEL_DIR, "ecg_preprocess.py")
    with open(preprocess_path, "w", encoding="utf-8") as f:   # FIX: utf-8
        f.write(preprocess_code)
    print(f"Preprocessing module saved -> {preprocess_path}")
    print("\nDone!  Normal | AFib | VFib -- all 3 classes ready for production.")


if __name__ == "__main__":
    train()