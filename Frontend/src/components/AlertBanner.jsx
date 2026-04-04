// Shows the combined alert level derived from both models
export default function AlertBanner({ alert }) {
  if (!alert) return null;

  const icons = { critical: "🚨", warning: "⚠️", normal: "✓" };

  return (
    <div style={{
      padding:      "12px 16px",
      borderRadius: 10,
      background:   `${alert.color}12`,
      border:       `1px solid ${alert.color}40`,
      display:      "flex",
      alignItems:   "center",
      gap:          10,
      marginBottom: 16,
    }}>
      <span style={{ fontSize: 18 }}>{icons[alert.level]}</span>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700, color: alert.color, textTransform: "capitalize" }}>
          {alert.level}
        </div>
        <div style={{ fontSize: 12, color: "#9ca3af" }}>{alert.message}</div>
      </div>
    </div>
  );
}
