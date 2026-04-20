import React from "react";

function ResultCard({ result }) {
  if (result === null || result === undefined) return null;

  const isGood = result.is_good_wine;
  const confidence = result.raw_score || 0;
  const percent = Math.round(confidence * 100);

  const color = isGood ? "#10b981" : "#64748b";

  return (
    <div className="result-card luxury" style={{ textAlign: "center", padding: "20px" }}>
      
      <p style={{ color: "#94a3b8", fontSize: "12px", letterSpacing: "2px" }}>
        ANALYSIS RESULT
      </p>

      {/*  Gauge */}
      <div style={{
        width: "120px",
        height: "120px",
        margin: "20px auto",
        borderRadius: "50%",
        background: `conic-gradient(${color} ${percent}%, #e2e8f0 ${percent}%)`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}>
        <div style={{
          width: "85%",
          height: "85%",
          background: "#fff",
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontWeight: "bold",
          fontSize: "22px"
        }}>
          {percent}%
        </div>
      </div>

      {/*  Nhận xét */}
      <h2 style={{ color: color }}>
        {isGood ? "Good Wine " : "Normal Wine"}
      </h2>

      {/*  Mô tả */}
      <p style={{ fontSize: "14px", color: "#475569" }}>
        {isGood
          ? "High-quality wine with balanced composition and strong characteristics."
          : "Wine is within acceptable quality range for normal consumption."}
      </p>

    </div>
  );
}

export default ResultCard;