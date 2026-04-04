import { useRef, useEffect } from "react";

export default function ECGChart({ signal, flagged, color = "#00d4aa" }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!signal?.length) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    const dpr    = window.devicePixelRatio || 1;
    const W      = canvas.offsetWidth;
    const H      = canvas.offsetHeight;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    // Background grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth   = 0.5;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    const min   = Math.min(...signal);
    const max   = Math.max(...signal);
    const range = (max - min) || 1;
    const step  = W / (signal.length - 1);
    const pad   = 20;
    const toY   = (v) => pad + ((max - v) / range) * (H - pad * 2);

    // Flagged segment highlight (anomaly zone)
    if (flagged) {
      const x1 = flagged[0] * step;
      const x2 = flagged[1] * step;
      ctx.fillStyle = "rgba(255,77,77,0.1)";
      ctx.fillRect(x1, 0, x2 - x1, H);
      ctx.strokeStyle = "rgba(255,77,77,0.5)";
      ctx.lineWidth   = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, 0); ctx.lineTo(x2, H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,77,77,0.85)";
      ctx.font      = "11px monospace";
      ctx.fillText("ANOMALY", x1 + 4, 14);
    }

    // ECG waveform
    ctx.beginPath();
    ctx.moveTo(0, toY(signal[0]));
    for (let i = 1; i < signal.length; i++) ctx.lineTo(i * step, toY(signal[i]));
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 5;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [signal, flagged, color]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "200px", display: "block" }} />;
}
