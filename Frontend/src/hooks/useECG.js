import { useState, useCallback } from "react";

const API = "http://localhost:8000";

export function useECG() {
  const [signal,   setSignal]   = useState(null);
  const [fileName, setFileName] = useState("");
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");

  // Parse uploaded .csv or .json file into flat number array
  const parseFile = useCallback((file) => {
    setFileName(file.name);
    setResult(null);
    setError("");

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result.trim();
        let parsed;
        if (file.name.endsWith(".json")) {
          parsed = JSON.parse(text);
          if (!Array.isArray(parsed)) throw new Error("JSON must be an array of numbers");
        } else {
          // CSV: single row or single column of floats
          parsed = text.split(/[\n,\r]+/).map(Number).filter((n) => !isNaN(n));
        }
        if (parsed.length < 50) throw new Error("Signal too short");
        setSignal(parsed);
      } catch (err) {
        setError("Parse error: " + err.message);
      }
    };
    reader.readAsText(file);
  }, []);

  // Load a synthetic demo NSR signal
  const loadDemo = useCallback(() => {
    const sig = [];
    for (let b = 0; b < 6; b++) {
      for (let t = 0; t < 187; t++) {
        const qrs = Math.exp(-0.5 * Math.pow((t - 60) / 5, 2)) * 1.0;
        const p   = Math.exp(-0.5 * Math.pow((t - 30) / 8, 2)) * 0.25;
        const t_w = Math.exp(-0.5 * Math.pow((t - 100) / 15, 2)) * 0.15;
        sig.push(Math.max(0, qrs + p + t_w + (Math.random() - 0.5) * 0.03));
      }
    }
    const max = Math.max(...sig);
    setSignal(sig.map((v) => v / max));
    setFileName("demo_nsr.json");
    setResult(null);
    setError("");
  }, []);

  // Send signal to FastAPI and get both model results
  const analyze = useCallback(async () => {
    if (!signal) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ signal, fs: 360 }),
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg);
      }
      const data = await res.json();
      // Update signal with preprocessed version from backend
      if (data.processed_signal?.length) setSignal(data.processed_signal);
      setResult(data);
    } catch (e) {
      setError("API error: " + e.message);
    } finally {
      setLoading(false);
    }
  }, [signal]);

  return { signal, fileName, result, loading, error, parseFile, loadDemo, analyze };
}
