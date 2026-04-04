import { useEffect, useState } from "react";

// One reusable card — used for BOTH rhythm result and disease result
export default function ResultCard({ title, result, accentColor, borderColor }) {
  const [barWidths, setBarWidths] = useState({});

  useEffect(() => {
    if (!result) return;
    // Animate bars in after mount
    const t = setTimeout(() => setBarWidths(result.probabilities), 120);
    return () => clearTimeout(t);
  }, [result]);

  if (!result) return null;

  const probColors = [accentColor, "#f5c542", "#ff4d4d"];

  return (
    <div style={{
      background:   "rgba(255,255,255,0.02)",
      border:       `1px solid ${borderColor}`,
      borderRadius: 14,
      padding:      20,
    }}>
      {/* Header */}
      <div style={{ fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 14 }}>
        {title}
      </div>

      {/* Diagnosis badge */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{
          background:   `${accentColor}18`,
          border:       `1.5px solid ${accentColor}`,
          borderRadius: "50%",
          width: 36, height: 36,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16,
        }}>
          {result.prediction === 0 ? "✓" : "⚠"}
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: accentColor }}>{result.label}</div>
          <div style={{ fontSize: 12, color: "#6b7280" }}>Confidence: {result.confidence}%</div>
        </div>
      </div>

      {/* Probability bars */}
      {Object.entries(result.probabilities).map(([label, pct], i) => (
        <div key={label} style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 12, color: "#9ca3af" }}>{label}</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: probColors[i] || accentColor, fontFamily: "monospace" }}>
              {pct}%
            </span>
          </div>
          <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.06)" }}>
            <div style={{
              height:     "100%",
              borderRadius: 3,
              background: probColors[i] || accentColor,
              width:      `${barWidths[label] ?? 0}%`,
              transition: "width 0.8s cubic-bezier(0.34,1.2,0.64,1)",
              boxShadow:  `0 0 6px ${probColors[i] || accentColor}60`,
            }} />
          </div>
        </div>
      ))}
    </div>
  );
}
