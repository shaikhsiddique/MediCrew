import numpy as np
import pandas as pd
import ast
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Flatten, Dense
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("🚀 Training 1D-CNN model...")

CLASS_NAMES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Ventricular Fibrillation"
]

# ───────────────────────────────────────────────────────────────
# FIX 1: resample_to_187 — replaces fix_length()
#
# OLD: fix_length() truncated to first 187 samples = only 0.37 s
#      of a 5-second signal. The CNN never saw a full rhythm.
#      Zero-padding made the rest of the signal flat zeros.
#
# NEW: interpolate the full signal down to 187 points, identical
#      to what prepare_model_input() does in main.py at inference.
#      Train/inference now see the same shape of signal.
# ───────────────────────────────────────────────────────────────
def resample_to_187(signal, target_len=187):
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(signal) == target_len:
        return signal.astype(np.float32)
    if len(signal) < 2:
        return np.repeat(signal.astype(np.float32), target_len)
    x_old = np.arange(len(signal), dtype=np.float64)
    x_new = np.linspace(0, len(signal) - 1, target_len)
    return np.interp(x_new, x_old, signal).astype(np.float32)


# ───────────────────────────────────────────────────────────────
# FIX 2: z-score normalization — must match inference
#
# OLD: no normalization applied during training.
# NEW: (x - mean) / std matches prepare_model_input() in main.py.
#      Without this, the model trains on raw amplitude values but
#      gets z-scored inputs at runtime — completely different scale.
# ───────────────────────────────────────────────────────────────
def normalize(signal):
    signal = np.asarray(signal, dtype=np.float32)
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def prepare(signal):
    return normalize(resample_to_187(signal))


# ───────────────────────────────────────────────────────────────
# FIX 3: ECG-aware augmentation — replaces SMOTE
#
# OLD: SMOTE interpolated between pairs of signals to generate
#      synthetic samples. For VFib (chaotic oscillations), averaging
#      two chaotic signals creates blurry patterns that look like
#      neither VFib nor AFib — the CNN learned a fuzzy boundary.
#
# NEW: physiologically realistic transforms:
#      - amplitude scaling  (simulates lead placement variation)
#      - gaussian noise     (simulates electrode noise)
#      - baseline wander    (simulates breathing artifact)
#      - time shift / roll  (simulates different R-peak alignment)
#      Each augmented signal is re-normalized after transforms.
# ───────────────────────────────────────────────────────────────
def augment_signal(x, rng):
    x = x.copy()
    x *= rng.uniform(0.85, 1.15)
    x += rng.normal(0, rng.uniform(0.01, 0.04), size=x.shape)
    freq  = rng.uniform(0.1, 0.5)
    phase = rng.uniform(0, 2 * np.pi)
    amp   = rng.uniform(0.01, 0.05)
    t     = np.linspace(0, 1, len(x))
    x    += amp * np.sin(2 * np.pi * freq * t + phase)
    x     = np.roll(x, rng.integers(-10, 11))
    return normalize(x)


def oversample_with_augmentation(X, y, seed=42):
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    majority_count = counts.most_common(1)[0][1]

    X_out, y_out = [X.copy()], [y.copy()]
    for cls, n_have in counts.items():
        n_need = majority_count - n_have
        if n_need <= 0:
            continue
        print(f"  class {cls}: {n_have} real → +{n_need} augmented")
        idx     = np.where(y == cls)[0]
        chosen  = idx[rng.integers(0, n_have, size=n_need)]
        aug     = np.stack([augment_signal(X[i], rng) for i in chosen])
        X_out.append(aug)
        y_out.append(np.full(n_need, cls, dtype=y.dtype))

    X_all = np.concatenate(X_out)
    y_all = np.concatenate(y_out)
    perm  = rng.permutation(len(X_all))
    return X_all[perm], y_all[perm]


# ───────────────────────────────────────────────────────────────
# LOAD + PREPROCESS
# ───────────────────────────────────────────────────────────────
def load_data(path: str):
    print("📂 Loading dataset...")

    df = pd.read_csv(path)

    label_map = {"Normal": 0, "AFib": 1, "VFib": 2}
    df = df[df["label"].isin(label_map)].reset_index(drop=True)

    # FIX 1 + 2: resample then z-score — matches inference pipeline
    X = np.stack([prepare(np.array(ast.literal_eval(s))) for s in df["signal"]])
    y = df["label"].map(label_map).values

    print(f"✅ Signal shape after fix: {X.shape}")
    print(f"📊 Original distribution: {Counter(y.tolist())}")

    return X, y


# ───────────────────────────────────────────────────────────────
# BUILD 1D-CNN MODEL  (unchanged)
# ───────────────────────────────────────────────────────────────
def build_cnn(input_len=187, num_classes=3):
    model = Sequential([

        # Block 1 — detect low-level wave features (P-wave, QRS onset)
        Conv1D(filters=32, kernel_size=5, activation="relu", padding="same",
               input_shape=(input_len, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Block 2 — detect mid-level patterns (QRS complex, T-wave)
        Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Block 3 — detect high-level rhythm patterns
        Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Classifier head
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ───────────────────────────────────────────────────────────────
# TRAIN
# ───────────────────────────────────────────────────────────────
def train():
    X, y = load_data("data/ecg_dataset_test.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nBefore augmentation: {Counter(y_train.tolist())}")

    # FIX 3: augmentation instead of SMOTE
    X_train, y_train = oversample_with_augmentation(X_train, y_train)

    print(f"After augmentation:  {Counter(y_train.tolist())}")
    print(f"Test dist:           {Counter(y_test.tolist())}")

    # Reshape for CNN → (samples, timesteps, channels)
    X_train_cnn = X_train.reshape(-1, 187, 1)
    X_test_cnn  = X_test.reshape(-1, 187, 1)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat  = to_categorical(y_test,  num_classes=3)

    # Build model
    model = build_cnn(input_len=187, num_classes=3)
    model.summary()

    # FIX 4: monitor val_loss not val_accuracy
    # val_accuracy stays high even when VFib recall quietly drops
    # because Normal + AFib dominate the test set numerically.
    # val_loss catches VFib degradation that accuracy misses.
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath="models/model_mitbih_cnn.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n⚡ Training CNN...")
    history = model.fit(
        X_train_cnn, y_train_cat,
        validation_split=0.15,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # ── EVALUATION ──
    y_pred_prob = model.predict(X_test_cnn)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    acc         = accuracy_score(y_test, y_pred)

    print("\n==============================")
    print("📊 MODEL PERFORMANCE")
    print("==============================")
    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    print("📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # FIX 4: confusion matrix with labels so VFib recall is obvious
    print("📉 Confusion Matrix (rows=true, cols=pred):")
    print("              Normal  AFib  VFib")
    cm = confusion_matrix(y_test, y_pred)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:6]:6s}  {row}")

    # Save final model
    model.save("models/model_mitbih_cnn.keras")
    print("\n💾 Model saved → models/model_mitbih_cnn.keras")


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()