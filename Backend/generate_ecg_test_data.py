import numpy as np
import pandas as pd
import json

# ── Step 1: Generate ECG-like signal ──
length = 187
t = np.linspace(0, 1, length)

# Simulated ECG waveform (sine + noise)
signal = 0.6 * np.sin(2 * np.pi * 5 * t) + 0.15 * np.random.randn(length)

# ── Step 2: Save as CSV ──
df = pd.DataFrame([signal])
csv_path = "test_ecg.csv"
df.to_csv(csv_path, index=False, header=False)

print(f"✅ CSV saved → {csv_path}")

# ── Step 3: Convert to JSON (API format) ──
json_data = {
    "signal": signal.tolist(),
    "fs": 360
}

json_path = "test_ecg.json"
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ JSON saved → {json_path}")

# ── Step 4: Print preview ──
print("\n📦 Sample JSON:")
print(json.dumps(json_data, indent=2)[:500])  # preview first part