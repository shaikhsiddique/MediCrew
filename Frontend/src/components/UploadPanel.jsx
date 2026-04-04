import { useRef, useState } from "react";

export default function UploadPanel({ onFile, onDemo, onAnalyze, loading, signal, fileName, error }) {
  const fileRef  = useRef(null);
  const [drag, setDrag] = useState(false);

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current.click()}
        style={{
          border:       `1.5px dashed ${drag ? "#00d4aa" : "rgba(255,255,255,0.12)"}`,
          borderRadius: 10,
          padding:      "32px 16px",
          textAlign:    "center",
          cursor:       "pointer",
          background:   drag ? "rgba(0,212,170,0.04)" : "transparent",
          transition:   "all 0.2s",
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.json"
          style={{ display: "none" }}
          onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
        />
        <div style={{ fontSize: 28, marginBottom: 8 }}>📂</div>
        <div style={{ fontSize: 13, color: "#9ca3af" }}>Drop .csv or .json file</div>
        <div style={{ fontSize: 11, color: "#4b5563", marginTop: 4 }}>or click to browse</div>
      </div>

      {/* File name */}
      {fileName && (
        <div style={{
          fontSize:   12, color: "#00d4aa",
          background: "rgba(0,212,170,0.06)",
          padding:    "6px 10px", borderRadius: 6,
          fontFamily: "monospace",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          ✓ {fileName}
        </div>
      )}

      {/* Demo button */}
      <button onClick={onDemo} style={{
        width: "100%", padding: "9px 0",
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8, color: "#9ca3af", fontSize: 13, cursor: "pointer",
      }}>
        Load demo signal
      </button>

      {/* Analyze button */}
      <button
        onClick={onAnalyze}
        disabled={!signal || loading}
        style={{
          width:        "100%", padding: "10px 0",
          background:   signal && !loading ? "#00d4aa" : "rgba(0,212,170,0.15)",
          border:       "none", borderRadius: 8,
          color:        signal && !loading ? "#0a0d12" : "#4b5563",
          fontSize:     14, fontWeight: 700,
          cursor:       signal ? "pointer" : "not-allowed",
          boxShadow:    signal && !loading ? "0 0 20px rgba(0,212,170,0.25)" : "none",
          transition:   "all 0.2s",
        }}
      >
        {loading ? "Analyzing…" : "Run Analysis →"}
      </button>

      {/* Error */}
      {error && (
        <div style={{
          padding: "8px 12px", borderRadius: 6, fontSize: 12,
          background: "rgba(255,77,77,0.08)",
          border:     "1px solid rgba(255,77,77,0.2)",
          color:      "#ff4d4d",
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
