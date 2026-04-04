import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal: np.ndarray, fs: int = 360, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def remove_baseline_wander(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(4, 0.5 / nyq, btype="high")
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray) -> np.ndarray:
    min_val = signal.min()
    max_val = signal.max()
    if max_val - min_val == 0:
        return signal
    return (signal - min_val) / (max_val - min_val)


def preprocess_signal(raw: list, fs: int = 360) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Remove baseline wander (high-pass)
    2. Bandpass filter (0.5 - 40 Hz)
    3. Normalize to [0, 1]
    4. Pad or trim to exactly 187 samples
    """
    sig = np.array(raw, dtype=float)

    if len(sig) > 187:
        # Find R-peak and extract window around it
        peaks, _ = find_peaks(sig, distance=50)
        if len(peaks) > 0:
            center = peaks[np.argmax(sig[peaks])]
            start = max(0, center - 93)
            end = start + 187
            if end > len(sig):
                start = len(sig) - 187
                end = len(sig)
            sig = sig[start:end]
        else:
            sig = sig[:187]

    sig = remove_baseline_wander(sig, fs)
    sig = bandpass_filter(sig, fs)
    sig = normalize(sig)

    # Pad with zeros if shorter than 187
    if len(sig) < 187:
        sig = np.pad(sig, (0, 187 - len(sig)))

    return sig


def get_flagged_segment(signal: np.ndarray) -> list:
    """Find the most anomalous segment for explainability highlight."""
    window = 30
    best_score = -1
    best_segment = [0, window]

    for i in range(0, len(signal) - window, window // 2):
        seg = signal[i:i + window]
        score = float(np.std(seg) * np.max(np.abs(np.diff(seg))))
        if score > best_score:
            best_score = score
            best_segment = [i, i + window]

    return best_segment
